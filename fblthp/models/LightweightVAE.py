import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding Module (unchanged)
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self.compute_positional_encodings(embed_dim, max_len))
        
    def compute_positional_encodings(self, embed_dim, max_len):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].permute(1, 0, 2)
        return x

def debug_tensor(name, tensor):
    """Print debug information about a tensor"""
    if tensor is None:
        print(f"DEBUG {name}: None")
        return
        
    print(f"DEBUG {name}:")
    print(f"  - Shape: {tensor.shape}")
    print(f"  - Type: {tensor.dtype}")
    print(f"  - Min: {tensor.min().item() if tensor.numel() > 0 else 'N/A'}")
    print(f"  - Max: {tensor.max().item() if tensor.numel() > 0 else 'N/A'}")
    print(f"  - Mean: {tensor.mean().item() if tensor.numel() > 0 else 'N/A'}")
    print(f"  - Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  - Has Inf: {torch.isinf(tensor).any().item()}")

# Encoder (mostly unchanged)
class LightweightEncoder(nn.Module):
    def __init__(self, 
                 main_vocab_size=20000,
                 mana_vocab_size=20,
                 text_embed_dim=128, 
                 mana_embed_dim=16,
                 hidden_dim=256,
                 latent_dim=64,
                 init_logvar_bias=-1.0):
        super(LightweightEncoder, self).__init__()
        
        # Embeddings
        self.text_embedding = nn.Embedding(main_vocab_size, text_embed_dim, padding_idx=0)
        self.mana_embedding = nn.Embedding(mana_vocab_size, mana_embed_dim, padding_idx=0)
        
        # Encoders
        self.name_encoder = nn.GRU(text_embed_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.type_encoder = nn.GRU(text_embed_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.flavor_encoder = nn.GRU(text_embed_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.mana_encoder = nn.GRU(mana_embed_dim, hidden_dim//4, batch_first=True)
        self.oracle_encoder = nn.GRU(text_embed_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        
        # Stats encoder
        self.stats_encoder = nn.Linear(4, hidden_dim//4)
        
        # Latent projections
        total_hidden = (hidden_dim * 4) + (hidden_dim//4) + (hidden_dim//4)
        self.fc_mu = nn.Linear(total_hidden, latent_dim)
        self.fc_logvar = nn.Linear(total_hidden, latent_dim)

        with torch.no_grad():
            # Initialize to small negative values to start with small variances
            self.fc_logvar.bias.fill_(init_logvar_bias)
    
    def encode(self, tokens, encoder, use_mana_embedding=False):
        """Unified encoding function that switches embedding based on field type"""
        # Handle empty sequences
        if tokens.shape[0] == 0 or not (tokens != 0).any():
            return torch.zeros(1, encoder.hidden_size * (2 if hasattr(encoder, 'bidirectional') else 1), 
                              device=tokens.device)
        
        # Mask out padding
        mask = (tokens != 0).any(dim=1)
        if not mask.any():
            return torch.zeros(1, encoder.hidden_size * (2 if hasattr(encoder, 'bidirectional') else 1), 
                              device=tokens.device)
        
        valid_tokens = tokens[mask]
        
        # Select appropriate embedding
        if use_mana_embedding:
            embedded = self.mana_embedding(valid_tokens)
        else:
            embedded = self.text_embedding(valid_tokens)
        
        _, hidden = encoder(embedded)
        
        # Process output based on encoder type
        if hasattr(encoder, 'bidirectional') and encoder.bidirectional:
            # Bidirectional GRU (text fields)
            hidden = hidden.transpose(0, 1).contiguous().view(hidden.size(1), -1)
        else:
            # Unidirectional GRU (mana field)
            hidden = hidden.squeeze(0)
            
        return hidden
    
    def forward(self, data):
        # Text field encoding
        name_hidden = self.encode(data['name_tokens'], self.name_encoder)
        type_hidden = self.encode(data['type_tokens'], self.type_encoder)
        oracle_hidden = self.encode(data['oracle_tokens'], self.oracle_encoder)
        flavor_hidden = self.encode(data['flavor_tokens'], self.flavor_encoder)
        
        # Mana encoding with specialized embeddings
        mana_hidden = self.encode(data['mana_tokens'], self.mana_encoder, use_mana_embedding=True)
        
        # Stats encoding - standardized handling of missing values
        stats = torch.zeros(data['cmc'].size(0), 4, device=data['cmc'].device)
        stats[:, 0] = data['cmc']
        
        # Handle power/toughness/loyalty with -1 for non-standard values
        for i, stat in enumerate([data['power'], data['toughness'], data['loyalty']]):
            stats[:, i+1] = torch.clamp(stat, min=-1, max=20)  # Prevent extreme values
        
        stats_hidden = self.stats_encoder(stats)
        
        # Combine all features
        combined = torch.cat([
            name_hidden, type_hidden, oracle_hidden, 
            flavor_hidden, mana_hidden, stats_hidden
        ], dim=1)
        
        # Project to latent space
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar

# Decoder (mostly unchanged)
class LightweightDecoder(nn.Module):
    def __init__(self, 
                 main_vocab_size=20000,
                 name_type_vocab_size=2000,
                 mana_vocab_size=20,
                 text_embed_dim=128, 
                 hidden_dim=256,
                 latent_dim=64,
                 max_oracle_len=100,
                 max_flavor_len=100,
                 max_name_len=10,
                 max_type_len=10,
                 max_mana_len=6,
                 sos_token_id=1,
                 eos_token_id=2):
        super(LightweightDecoder, self).__init__()
        
        # Store config
        self.latent_dim = latent_dim
        self.text_embed_dim = text_embed_dim
        self.max_lengths = {
            'oracle': max_oracle_len,
            'flavor': max_flavor_len,
            'name': max_name_len,
            'type': max_type_len,
            'mana': max_mana_len
        }
        
        # Store token IDs as class variables for easier configuration
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
        # Task embeddings (0=oracle, 1=flavor, 2=name, 3=type)
        self.task_embedding = nn.Embedding(4, text_embed_dim)
        
        # Token embedding (shared across all text components)
        self.token_embedding = nn.Embedding(main_vocab_size, text_embed_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(text_embed_dim, max_len=max(max_oracle_len, max_flavor_len)+1)
        self.pos_encoder_small = PositionalEncoding(text_embed_dim, max_len=max(max_name_len, max_type_len)+1)
        
        # Projection from latent to memory
        self.latent_to_memory_large = nn.Linear(latent_dim, text_embed_dim)
        self.latent_to_memory_small = nn.Linear(latent_dim, text_embed_dim)
        
        # Large transformer for oracle and flavor text
        large_decoder_layer = nn.TransformerDecoderLayer(
            d_model=text_embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.large_transformer = nn.TransformerDecoder(
            large_decoder_layer, 
            num_layers=2
        )
        
        # Small transformer for name and type
        small_decoder_layer = nn.TransformerDecoderLayer(
            d_model=text_embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim//2,
            dropout=0.1,
            batch_first=True
        )
        self.small_transformer = nn.TransformerDecoder(
            small_decoder_layer, 
            num_layers=1
        )
        
        # Output projections
        self.oracle_output = nn.Linear(text_embed_dim, main_vocab_size)
        self.flavor_output = nn.Linear(text_embed_dim, main_vocab_size)
        self.name_output = nn.Linear(text_embed_dim, name_type_vocab_size)
        self.type_output = nn.Linear(text_embed_dim, name_type_vocab_size)
        
        # Mana cost generation
        self.mana_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, max_mana_len * mana_vocab_size)
        )
        
        # Stats regression
        self.stats_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, 4)  # [CMC, Power, Toughness, Loyalty]
        )
    
    def _create_causal_mask(self, size, device):
        """Create a causal mask for the transformer decoder"""
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask
    
    def forward(self, z, targets=None):
        """
        Decode from latent space with teacher forcing
        
        Args:
            z (Tensor): Latent vector [batch_size, latent_dim]
            targets (dict): Target sequences for teacher forcing
            
        Returns:
            dict: Dictionary of model outputs
        """
        batch_size = z.size(0)
        results = {}
        
        # Always use teacher forcing in this implementation
        if targets is None:
            raise ValueError("Teacher forcing is required - targets must be provided")
        
        # Generate stats through direct regression
        stats = self.stats_decoder(z)
        results['cmc'] = stats[:, 0]
        results['power'] = stats[:, 1]
        results['toughness'] = stats[:, 2]
        results['loyalty'] = stats[:, 3]
        
        # Generate mana cost (categorical distribution)
        mana_logits = self.mana_generator(z).view(
            batch_size, self.max_lengths['mana'], -1
        )
        results['mana_logits'] = mana_logits
        
        # Process text fields with transformers
        for field_idx, field in enumerate(['oracle', 'flavor', 'name', 'type']):
            # Select correct transformer and projections
            if field_idx < 2:  # oracle, flavor
                transformer = self.large_transformer
                memory_proj = self.latent_to_memory_large
                pos_encoder = self.pos_encoder
                max_len = self.max_lengths[field]
            else:  # name, type
                transformer = self.small_transformer
                memory_proj = self.latent_to_memory_small
                pos_encoder = self.pos_encoder_small
                max_len = self.max_lengths[field]
            
            # Select correct output projection
            if field == 'oracle':
                output_layer = self.oracle_output
            elif field == 'flavor':
                output_layer = self.flavor_output
            elif field == 'name':
                output_layer = self.name_output
            else:  # type
                output_layer = self.type_output
            
            # Get target sequence
            tgt = targets[f'{field}_tokens']
            
            # Ensure sequence isn't longer than max length
            if tgt.size(1) > max_len:
                tgt = tgt[:, :max_len]
            
            # Create task embedding
            task_embed = self.task_embedding(
                torch.full((batch_size,), field_idx, device=z.device)
            ).unsqueeze(1)  # [batch_size, 1, embed_dim]
            
            # Shift target for teacher forcing (remove last token, prepend SOS)
            input_ids = torch.cat([
                torch.full((batch_size, 1), self.sos_token_id, device=z.device),
                tgt[:, :-1]
            ], dim=1)
            
            # Embed input tokens
            token_embeds = self.token_embedding(input_ids)
            
            # Prepend task embedding
            decoder_input = torch.cat([task_embed, token_embeds], dim=1)
            
            # Add positional encoding
            pos_encoded = pos_encoder(decoder_input)
            
            # Project latent to memory
            memory = memory_proj(z).unsqueeze(1)
            memory = memory.repeat(1, pos_encoded.size(1), 1)
            
            # Create causal mask
            causal_mask = self._create_causal_mask(pos_encoded.size(1), z.device)
            
            # Apply transformer decoder with teacher forcing
            decoder_output = transformer(
                pos_encoded, 
                memory,
                tgt_mask=causal_mask
            )
            
            # Remove task token from output
            decoder_output = decoder_output[:, 1:]
            
            # Project to vocabulary
            logits = output_layer(decoder_output)
            
            # Store results
            results[f'{field}_logits'] = logits
        
        return results
    
    def generate(self, z):
        """Generate sequences from latent space (autoregressive generation at inference time)"""
        batch_size = z.size(0)
        results = {}
        
        # Generate stats
        stats = self.stats_decoder(z)
        results['cmc'] = stats[:, 0]
        results['power'] = stats[:, 1]
        results['toughness'] = stats[:, 2]
        results['loyalty'] = stats[:, 3]
        
        # Generate mana cost
        mana_logits = self.mana_generator(z).view(
            batch_size, self.max_lengths['mana'], -1
        )
        mana_tokens = torch.argmax(mana_logits, dim=-1)
        results['mana_tokens'] = mana_tokens
        results['mana_logits'] = mana_logits
        
        # Generate text fields autoregressively
        for field_idx, field in enumerate(['oracle', 'flavor', 'name', 'type']):
            # Select correct transformer and projections
            if field_idx < 2:  # oracle, flavor
                transformer = self.large_transformer
                memory_proj = self.latent_to_memory_large
                pos_encoder = self.pos_encoder
                max_len = self.max_lengths[field]
            else:  # name, type
                transformer = self.small_transformer
                memory_proj = self.latent_to_memory_small
                pos_encoder = self.pos_encoder_small
                max_len = self.max_lengths[field]
            
            # Select correct output projection
            if field == 'oracle':
                output_layer = self.oracle_output
            elif field == 'flavor':
                output_layer = self.flavor_output
            elif field == 'name':
                output_layer = self.name_output
            else:  # type
                output_layer = self.type_output
            
            # Create task embedding
            task_embed = self.task_embedding(
                torch.full((batch_size,), field_idx, device=z.device)
            ).unsqueeze(1)  # [batch_size, 1, embed_dim]
            
            # Start with SOS token
            start_token = torch.full((batch_size, 1), self.sos_token_id, 
                                    device=z.device)
            start_embed = self.token_embedding(start_token)
            
            # Combine task embedding and start token
            current_output = torch.cat([task_embed, start_embed], dim=1)
            
            # Project latent to memory
            memory = memory_proj(z).unsqueeze(1)
            
            # Output containers
            output_ids = torch.zeros(batch_size, max_len, dtype=torch.long, 
                                   device=z.device)
            output_logits = torch.zeros(batch_size, max_len, output_layer.out_features, 
                                      device=z.device)
            
            # Autoregressive generation
            for i in range(max_len):
                # Add positional encoding
                pos_encoded = pos_encoder(current_output)
                
                # Create causal mask
                causal_mask = self._create_causal_mask(pos_encoded.size(1), z.device)
                
                # Apply transformer decoder
                decoder_output = transformer(
                    pos_encoded, 
                    memory.repeat(1, pos_encoded.size(1), 1),
                    tgt_mask=causal_mask
                )
                
                # Get last token prediction
                last_token_feat = decoder_output[:, -1, :]
                
                # Project to vocabulary
                logits = output_layer(last_token_feat)
                
                # Store outputs
                output_logits[:, i] = logits
                
                # Sample next token (argmax for deterministic generation)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                output_ids[:, i] = next_token.squeeze(1)
                
                # Stop if EOS token or reached max length
                if i == max_len - 1 or (next_token == self.eos_token_id).all():
                    break
                
                # Prepare for next iteration
                next_token_embed = self.token_embedding(next_token)
                current_output = torch.cat([current_output, next_token_embed], dim=1)
            
            # Store results
            results[f'{field}_tokens'] = output_ids
            results[f'{field}_logits'] = output_logits
        
        return results


class LightweightVAE(nn.Module):
    def __init__(self, 
                 main_vocab_size=20000,
                 name_type_vocab_size=2000,
                 mana_vocab_size=20,
                 text_embed_dim=128, 
                 mana_embed_dim=16,
                 hidden_dim=256,
                 latent_dim=64,
                 max_oracle_len=100,
                 max_flavor_len=100,
                 max_name_len=10,
                 max_type_len=10,
                 max_mana_len=6,
                 kl_weight=0.1,
                 debug_mode=False):
        super(LightweightVAE, self).__init__()
        
        # Save parameters
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.debug_mode = debug_mode
        
        # Initialize encoder and decoder
        self.encoder = LightweightEncoder(
            main_vocab_size=main_vocab_size,
            mana_vocab_size=mana_vocab_size,
            text_embed_dim=text_embed_dim,
            mana_embed_dim=mana_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.decoder = LightweightDecoder(
            main_vocab_size=main_vocab_size,
            name_type_vocab_size=name_type_vocab_size,
            mana_vocab_size=mana_vocab_size,
            text_embed_dim=text_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_oracle_len=max_oracle_len,
            max_flavor_len=max_flavor_len,
            max_name_len=max_name_len,
            max_type_len=max_type_len,
            max_mana_len=max_mana_len
        )
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        
        Args:
            mu (Tensor): Mean of the latent Gaussian
            logvar (Tensor): Log variance of the latent Gaussian
            
        Returns:
            Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def compute_kl_divergence(self, mu, logvar):
        """
        Compute KL divergence between N(mu, var) and N(0, 1)
        Uses a numerically stable implementation
        
        Args:
            mu (Tensor): Mean of the latent Gaussian
            logvar (Tensor): Log variance of the latent Gaussian
            
        Returns:
            Tensor: KL divergence
        """
        # Clip values for numerical stability
        mu_clipped = torch.clamp(mu, min=-8.0, max=8.0)
        logvar_clipped = torch.clamp(logvar, min=-8.0, max=8.0)
        
        # Stable KL calculation
        kl_div = 0.5 * torch.sum(
            torch.exp(logvar_clipped) + mu_clipped.pow(2) - 1.0 - logvar_clipped
        ) / mu.size(0)
        
        # Optional debug information
        if self.debug_mode:
            with torch.no_grad():
                kl_terms = {
                    'var': torch.exp(logvar_clipped).mean().item(),
                    'mu_sq': mu_clipped.pow(2).mean().item(),
                    'logvar': logvar_clipped.mean().item()
                }
                print(f"KL Debug - var: {kl_terms['var']:.4f}, "
                      f"mu²: {kl_terms['mu_sq']:.4f}, "
                      f"logvar: {kl_terms['logvar']:.4f}, "
                      f"KL: {kl_div.item():.4f}")
        
        return kl_div
    
    def forward(self, data):
        """
        Forward pass through the VAE
        
        Args:
            data (dict): Dictionary containing input data
            
        Returns:
            dict: Dictionary containing reconstruction outputs and latent variables
        """
        # Encode
        mu, logvar = self.encoder(data)
        
        # Compute KL divergence (for debugging only)
        if self.debug_mode:
            self.compute_kl_divergence(mu, logvar)
        
        # Sample latent variable
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructions = self.decoder(z, targets=data)
        
        # Add latent variables to output
        reconstructions['mu'] = mu
        reconstructions['logvar'] = logvar
        reconstructions['z'] = z
        
        return reconstructions
    
    def compute_loss(self, data, reconstructions):
        """
        Compute VAE loss (reconstruction + KL divergence) with enhanced stability
        
        Args:
            data (dict): Dictionary containing input data
            reconstructions (dict): Dictionary containing reconstruction outputs
            
        Returns:
            Tensor: Total loss
            dict: Dictionary containing individual loss components
        """
        # Initialize loss components dictionary
        loss_components = {}
        
        # Compute KL divergence
        kl_loss = self.compute_kl_divergence(
            reconstructions['mu'], reconstructions['logvar']
        )
        loss_components['kl_loss'] = kl_loss
        
        # Simplified approach to compute total reconstruction loss
        total_recon_loss = 0.0
        valid_token_count = 0
        
        # Text field reconstruction losses
        for field in ['oracle', 'flavor', 'name', 'type']:
            # Get target tokens and reconstruction logits
            target = data[f'{field}_tokens']
            logits = reconstructions[f'{field}_logits']
            
            # Create mask for non-padding tokens
            mask = (target != 0).float()
            valid_tokens = mask.sum()
            
            if valid_tokens > 0:
                # Compute cross-entropy loss with masking
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = target.view(-1)
                
                field_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
                field_loss = field_loss.view_as(target) * mask
                field_loss = field_loss.sum() / valid_tokens
                
                # Store individual field loss
                loss_components[f'{field}_loss'] = field_loss
                
                # Add to total reconstruction loss
                total_recon_loss += field_loss
                valid_token_count += 1
        
        # Mana cost reconstruction loss
        mana_logits = reconstructions['mana_logits']
        mana_targets = data['mana_tokens']
        mana_mask = (mana_targets != 0).float()
        valid_mana_tokens = mana_mask.sum()
        
        if valid_mana_tokens > 0:
            flat_mana_logits = mana_logits.view(-1, mana_logits.size(-1))
            flat_mana_targets = mana_targets.view(-1)
            
            mana_loss = F.cross_entropy(flat_mana_logits, flat_mana_targets, reduction='none')
            mana_loss = mana_loss.view_as(mana_targets) * mana_mask
            mana_loss = mana_loss.sum() / valid_mana_tokens
            
            loss_components['mana_loss'] = mana_loss
            total_recon_loss += mana_loss
            valid_token_count += 1
        
        # --------- DEBUGGING MSE LOSS SECTION ---------
        # Stats regression losses (with careful handling for stability)
        stats_loss = 0.0
        stats_count = 0
        
        for stat in ['cmc', 'power', 'toughness', 'loyalty']:
            if stat in data and stat in reconstructions:
                # Debug the values 
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    debug_tensor(f"data[{stat}]", data[stat])
                    debug_tensor(f"reconstructions[{stat}]", reconstructions[stat])
                
                # Ensure predictions are finite
                pred_values = reconstructions[stat]
                if torch.isnan(pred_values).any() or torch.isinf(pred_values).any():
                    # Replace NaN/Inf with zeros to prevent propagation
                    pred_values = torch.nan_to_num(pred_values, nan=0.0, posinf=0.0, neginf=0.0)
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"WARNING: Found NaN/Inf in {stat} predictions, replaced with zeros")
                
                # Clamp prediction values to reasonable range
                pred_values = torch.clamp(pred_values, min=-20.0, max=20.0)
                
                # Only compute loss for valid values (>= 0)
                valid_mask = (data[stat] >= 0).float()
                valid_count = valid_mask.sum()
                
                if valid_count > 0:
                    # Get target values and ensure they're valid
                    target_values = data[stat] * valid_mask
                    
                    # Compute squared difference manually with careful handling
                    squared_diff = (pred_values * valid_mask - target_values).pow(2)
                    
                    # Check for NaN in squared differences
                    if torch.isnan(squared_diff).any():
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            debug_tensor(f"squared_diff_{stat}", squared_diff)
                        # Replace NaN with zeros
                        squared_diff = torch.nan_to_num(squared_diff, nan=0.0)
                    
                    # Sum and normalize
                    stat_loss = squared_diff.sum() / valid_count
                    
                    # Final safety check
                    if not torch.isnan(stat_loss) and not torch.isinf(stat_loss):
                        loss_components[f'{stat}_loss'] = stat_loss
                        stats_loss += stat_loss
                        stats_count += 1
                    else:
                        # Use a small constant loss if we still have NaN
                        stat_loss = torch.tensor(0.1, device=data[stat].device)
                        loss_components[f'{stat}_loss'] = stat_loss
                        stats_loss += stat_loss
                        stats_count += 1
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            print(f"WARNING: Final {stat} loss was NaN, using constant")
        
        # Add stats loss to total reconstruction loss
        if stats_count > 0:
            total_recon_loss += stats_loss / stats_count
        
        # Average the reconstruction loss
        if valid_token_count > 0:
            total_recon_loss /= valid_token_count
        
        # Store total reconstruction loss
        loss_components['recon_loss'] = total_recon_loss
        
        # Compute total loss
        total_loss = total_recon_loss + self.kl_weight * kl_loss
        loss_components['total_loss'] = total_loss
        
        # Final sanity check
        if torch.isnan(total_loss):
            print("WARNING: Total loss is NaN!")
            # Return a small constant loss to avoid crashing
            return torch.tensor(1.0, device=total_loss.device, requires_grad=True), loss_components
        
        return total_loss, loss_components
    
    @property
    def device(self):
        """Get the device the model is on"""
        return next(self.parameters()).device
    
    def set_debug_mode(self, enabled=True):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
        return self