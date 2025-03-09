import torch
import torch.nn as nn
import math
from transformers import BertModel
from transformers import GPT2LMHeadModel, GPT2Config

# Check if a GPU is available and use it if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        
        # Register pe as a buffer, meaning it won't be updated during training
        self.register_buffer('pe', self.compute_positional_encodings(embed_dim, max_len))
    def compute_positional_encodings(self, embed_dim, max_len):
        # Create a matrix of shape (max_len, embed_dim) for positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute positional encodings using sine and cosine functions
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe
    def forward(self, x):
        # Add positional encodings to the input embeddings
        x = x + self.pe[:x.size(1), :].permute(1, 0, 2)
        return x






class DecomposedEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim, num_heads, mlp_dim, num_layers, max_len=125, dropout_rate=0.2):
        super(DecomposedEncoder, self).__init__()
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len + 2)

        #Card Name Encoder
        card_name_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True, dropout=dropout_rate
        )
        self.card_name_query = nn.Parameter(torch.randn(embed_dim))
        self.card_name_encoder = nn.TransformerEncoder(card_name_layers, num_layers=num_layers)
        
        #Mana Cost Encoder
        mana_cost_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True, dropout=dropout_rate
        )
        self.mana_query = nn.Parameter(torch.randn(embed_dim))
        self.mana_encoder = nn.TransformerEncoder(mana_cost_layers, num_layers=num_layers)

        #Card Type & Subtype Encoder
        card_type_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True, dropout=dropout_rate
        )
        self.card_type_query = nn.Parameter(torch.randn(embed_dim))
        self.card_type_encoder = nn.TransformerEncoder(card_type_layers, num_layers=num_layers)
        
        #Oracle Text Encoder
        oracle_text_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True, dropout=dropout_rate
        )
        self.oracle_text_encoder = nn.TransformerEncoder(oracle_text_layers, num_layers=num_layers)
        self.oracle_text_query = nn.Parameter(torch.randn(embed_dim))

        #Flavor Text Encoder
        flavor_text_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True, dropout=dropout_rate
        )
        self.flavor_text_encoder = nn.TransformerEncoder(flavor_text_layers, num_layers=num_layers)
        self.flavor_text_query = nn.Parameter(torch.randn(embed_dim))
        

        #Power Embedding (None, X, Y, *, 0 ... 20)
        self.power_embedding = nn.Embedding(25, embed_dim)

        #Toughness Embedding (None, X, Y, *, 0 ... 20)
        self.toughness_embedding = nn.Embedding(25, embed_dim)

        #Loyalty Embedding (None, X, 0 ... 10)
        self.loyalty_embedding = nn.Embedding(13, embed_dim)
        
        # Dropout after the embedding layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Softmax for attention pooling
        
        self.softmax = nn.Softmax(dim=-1)
        
        # Linear layers for mean and log variance
        self.fc_mu = nn.Linear(embed_dim, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)  # Log variance

    def forward(self, x):
        # Assuming x is a dictionary with the following structure:
        # {
        #     "card_name": tensor,  # Shape: [batch_size, sequence_length]
        #     "mana_cost": tensor,  # Shape: [batch_size, sequence_length]
        #     "card_type": tensor,  # Shape: [batch_size, sequence_length]
        #     "oracle_text": tensor,  # Shape: [batch_size, sequence_length]
        #     "flavor_text": tensor,  # Shape: [batch_size, sequence_length]
        #     "power": tensor,  # Shape: [batch_size, 1]
        #     "toughness": tensor,  # Shape: [batch_size, 1]
        #     "loyalty": tensor  # Shape: [batch_size, 1]
        # }


        # If x is 1D, add a batch dimension
        for attribute in x:
            if len(x[attribute].shape) == 1:
                x[attribute] = x[attribute].unsqueeze(0)  # Shape: [batch_size, sequence_length]
        
        # Embedding tokens into continuous space
        name_embedded = self.embedding(x["card_name"])  # Shape: [batch_size, sequence_length, embed_dim]
        mana_embedded = self.embedding(x["mana_cost"])  # Shape: [batch_size, sequence_length, embed_dim]
        type_embedded = self.embedding(x["card_type"])  # Shape: [batch_size, sequence_length, embed_dim]
        oracle_embedded = self.embedding(x["oracle_text"])  # Shape: [batch_size, sequence_length, embed_dim]
        flavor_embedded = self.embedding(x["flavor_text"])  # Shape: [batch_size, sequence_length, embed_dim]
        
        #Apply dropout to embeddings
        name_embedded = self.dropout(name_embedded)
        mana_embedded = self.dropout(mana_embedded)
        type_embedded = self.dropout(type_embedded)
        oracle_embedded = self.dropout(oracle_embedded)
        flavor_embedded = self.dropout(flavor_embedded)

        # Add positional encodings
        name_embedded = self.pos_encoder(name_embedded)
        mana_embedded = self.pos_encoder(mana_embedded)
        type_embedded = self.pos_encoder(type_embedded)
        oracle_embedded = self.pos_encoder(oracle_embedded)
        flavor_embedded = self.pos_encoder(flavor_embedded)
        
        # Pass through respective TransformerEncoder layers
        name_encoded = self.card_name_encoder(name_embedded)
        mana_encoded = self.mana_encoder(mana_embedded)
        type_encoded = self.card_type_encoder(type_embedded)
        oracle_encoded = self.oracle_text_encoder(oracle_embedded)
        flavor_encoded = self.flavor_text_encoder(flavor_embedded)

        #Attention-based pooling for respective encodings
        name_attention_scores = torch.matmul(name_encoded, self.card_name_query)
        mana_attention_scores = torch.matmul(mana_encoded, self.mana_query)
        type_attention_scores = torch.matmul(type_encoded, self.card_type_query)
        oracle_attention_scores = torch.matmul(oracle_encoded, self.oracle_text_query)
        flavor_attention_scores = torch.matmul(flavor_encoded, self.flavor_text_query)

        name_attention_weights = self.softmax(name_attention_scores)
        mana_attention_weights = self.softmax(mana_attention_scores)
        type_attention_weights = self.softmax(type_attention_scores)
        oracle_attention_weights = self.softmax(oracle_attention_scores)
        flavor_attention_weights = self.softmax(flavor_attention_scores)

        name_attention_weights = name_attention_weights.unsqueeze(-1)
        mana_attention_weights = mana_attention_weights.unsqueeze(-1)
        type_attention_weights = type_attention_weights.unsqueeze(-1)
        oracle_attention_weights = oracle_attention_weights.unsqueeze(-1)
        flavor_attention_weights = flavor_attention_weights.unsqueeze(-1)

        name_pooled = torch.sum(name_encoded * name_attention_weights, dim=1)
        mana_pooled = torch.sum(mana_encoded * mana_attention_weights, dim=1)
        type_pooled = torch.sum(type_encoded * type_attention_weights, dim=1)
        oracle_pooled = torch.sum(oracle_encoded * oracle_attention_weights, dim=1)
        flavor_pooled = torch.sum(flavor_encoded * flavor_attention_weights, dim=1)

        #Fetch power, toughness, and loyalty embeddings
        power_embedded = self.power_embedding(x["power"])  # Shape: [batch_size, 1, embed_dim]
        toughness_embedded = self.toughness_embedding(x["toughness"])
        loyalty_embedded = self.loyalty_embedding(x["loyalty"])

        #Just add them all together for now (considering attention-based pooling in the future)
        pooled = name_pooled + mana_pooled + type_pooled + oracle_pooled + flavor_pooled + power_embedded + toughness_embedded + loyalty_embedded

        
        # Compute mean and log variance for reparameterization
        mu = self.fc_mu(pooled)  # Shape: [batch_size, embed_dim]
        logvar = self.fc_logvar(pooled)  # Shape: [batch_size, embed_dim]
        
        return mu, logvar

class SmallAttributesDecoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, dropout_rate=0.1):
        super(SmallAttributesDecoder, self).__init__()
        self.shared_features = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.power_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 25)
        )

        self.toughness_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 25)
        )

        self.loyalty_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 13)
        )

        self.has_pt = nn.Linear(embed_dim, 1)
        self.has_loyalty = nn.Linear(embed_dim, 1)

    
    def forward(self, latents):

        if len(latents.shape) == 1:
            latents = latents.unsqueeze(0)
        elif len(latents.shape) >= 3:
            raise ValueError("Latents must be 1D or 2D tensor")
        
        features = self.shared_features(latents)

        has_pt = torch.sigmoid(self.has_pt(features))
        has_loyalty = torch.sigmoid(self.has_pt(features))

        power_logits = self.power_head(features)
        toughness_logits = self.toughness_head(features)
        loyalty_logits = self.loyalty_head(features)

        return {
            'power' : power_logits,
            'toughness' : toughness_logits,
            'loyalty': loyalty_logits,
            'has_pt': has_pt, 
            'has_loyalty': has_loyalty,
        }




class AttributeDecomposedDecoder(nn.Module):
    def __init__(self, vocab_size, mana_vocab_size, latent_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len=125, dropout_rate=0.1):
        super(AttributeDecomposedDecoder, self).__init__()
        self.output_dim = vocab_size
        
        # Embedding layer for the output tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Embedding layer for the mana tokens
        self.mana_embedding = nn.Embedding(mana_vocab_size, embed_dim)
        
        # Positional encoding for the decoder
        self.pos_encoder = PositionalEncoding(embed_dim, max_len + 2)


        #Power, Toughness, Loyalty Predictors
        self.small_decoder = SmallAttributesDecoder(latent_dim, embed_dim, dropout_rate)


        self.name_predictor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout_rate
            ), num_layers=num_layers
        )
        self.name_out = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout before the final layer
            nn.Linear(embed_dim, vocab_size)
        )


        self.mana_predictor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout_rate
            ), num_layers=num_layers
        )
        self.mana_out = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout before the final layer
            nn.Linear(embed_dim, vocab_size)
        )


        self.type_predictor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout_rate
            ), num_layers=num_layers
        )
        self.type_out = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout before the final layer
            nn.Linear(embed_dim, vocab_size)
        )


        self.oracle_predictor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout_rate
            ), num_layers=num_layers
        )
        self.oracle_out = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout before the final layer
            nn.Linear(embed_dim, vocab_size)
        )


        self.flavor_predictor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout_rate
            ), num_layers=num_layers
        )
        self.flavor_out = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout before the final layer
            nn.Linear(embed_dim, vocab_size)
        )


        
        self.tgt_mask = torch.triu(torch.ones(max_len, max_len, device=device) * float('-inf'), diagonal=1)
        
        # Dropout for embeddings and positional encodings
        self.dropout = nn.Dropout(dropout_rate)
        
        # Store maximum length and special tokens
        self.max_len = max_len
        self.sos_token = 1
        self.eos_token = 2

    def forward(self, latent, target_seq):
        """
        Forward pass with 100% teacher forcing.
        
        Args:
            memory: Encoder outputs, shape [batch_size, memory_dim].
            target_seq: Ground truth target sequence, dictionary.
        
        Returns:
            output_logits: Predicted logits for all steps, shape [batch_size, seq_len, output_dim].
        """
        device = latent.device
        # Target Sequence
        # {
        #     "card_name": tensor,  # Shape: [batch_size, sequence_length]
        #     "mana_cost": tensor,  # Shape: [batch_size, sequence_length]
        #     "card_type": tensor,  # Shape: [batch_size, sequence_length]
        #     "oracle_text": tensor,  # Shape: [batch_size, sequence_length]
        #     "flavor_text": tensor,  # Shape: [batch_size, sequence_length]
        #     "power": tensor,  # Shape: [batch_size, 1]
        #     "toughness": tensor,  # Shape: [batch_size, 1]
        #     "loyalty": tensor  # Shape: [batch_size, 1]
        # }


        small_decoder_outputs = self.small_decoder(latent)
        power = small_decoder_outputs['power'] # Shape: [batch_size, 25]
        toughness = small_decoder_outputs['toughness'] # Shape: [batch_size, 25]
        loyalty = small_decoder_outputs['loyalty'] # Shape: [batch_size, 13]
        has_pt = small_decoder_outputs['has_pt'] # Shape: [batch_size, 1]
        has_loyalty = small_decoder_outputs['has_loyalty'] # Shape: [batch_size, 1]


        # Embed the entire target sequence (assumes <sos> is already prepended to target_seq)
        name_emb = self.dropout(self.pos_encoder(self.embedding(target_seq["name_tokens"])))  # Shape: [batch_size, seq_len, embed_dim]
        mana_emb = self.dropout(self.pos_encoder(self.embedding(target_seq["mana_tokens"])))  # Shape: [batch_size, seq_len, embed_dim]
        type_emb = self.dropout(self.pos_encoder(self.embedding(target_seq["type_line_tokens"])))  # Shape: [batch_size, seq_len, embed_dim]
        oracle_emb = self.dropout(self.pos_encoder(self.embedding(target_seq["oracle_tokens"])))  # Shape: [batch_size, seq_len, embed_dim]
        flavor_emb = self.dropout(self.pos_encoder(self.embedding(target_seq["flavor_text_tokens"])))  # Shape: [batch_size, seq_len, embed_dim]

        name = self.name_predictor(
            tgt=name_emb,
            memory=latent.unsqueeze(1),
            tgt_mask=self.tgt_mask[:name_emb.size(1), :name_emb.size(1)]
        )

        mana = self.mana_predictor(
            tgt=mana_emb,
            memory=latent.unsqueeze(1),
            tgt_mask=self.tgt_mask[:mana_emb.size(1), :mana_emb.size(1)]
        )

        card_type = self.type_predictor(
            tgt=type_emb,
            memory=latent.unsqueeze(1),
            tgt_mask=self.tgt_mask[:type_emb.size(1), :type_emb.size(1)]
        )

        oracle_text = self.oracle_predictor(
            tgt=oracle_emb,
            memory=latent.unsqueeze(1),
            tgt_mask=self.tgt_mask[:oracle_emb.size(1), :oracle_emb.size(1)]
        )

        flavor_text = self.flavor_predictor(
            tgt=flavor_emb,
            memory=latent.unsqueeze(1),
            tgt_mask=self.tgt_mask[:flavor_emb.shape(1), :flavor_emb.shape(1)]
        )

        # Compute output logits for all time steps
        name_logits = self.name_out(name)
        mana_logits = self.mana_out(mana)
        type_logits = self.type_out(card_type)
        oracle_logits = self.oracle_out(oracle_text)
        flavor_logits = self.flavor_out(flavor_text)



        return {
            "name_logits": name_logits,
            "mana_logits": mana_logits,
            "type_logits": type_logits,
            "oracle_logits": oracle_logits,
            "flavor_logits": flavor_logits,
            "power_logits": power,
            "toughness_logits": toughness,
            "loyalty_logits": loyalty,
            "has_pt": has_pt,
            "has_loyalty": has_loyalty
        }

class ADVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len=125, dropout=0.1):
        super(ADVAE, self).__init__()
        
        # Encoder (Pooling Transformer Encoder)
        #self.encoder = PoolingTransformerEncoder(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len)
        self.encoder = DecomposedEncoder(vocab_size, embed_dim, latent_dim, num_heads, hidden_dim, num_encoder_layers, max_len, dropout)
        
        # Decoder (Pooled Transformer Decoder)
        self.decoder = AttributeDecomposedDecoder(vocab_size, embed_dim, num_heads, hidden_dim, num_decoder_layers, max_len, dropout)

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random noise
        return mu + eps * std          # z = mu + eps * std

    def print_model_size(self, debug=False):
        param_size = 0
        params = 0
        if debug:
            print("\nParameters:")
        for name, param in self.named_parameters():
            size = param.nelement() * param.element_size()  # number of elements * size of each element
            param_size += size
            params += param.nelement()
            if size < 1024*10:
                continue
            if debug:
                print(f"  Name: {name}, Shape: {param.shape}, Memory: {size / (1024 ** 2):.2f} MB")

        buffer_size = 0
        buffers = 0
        for name, buffer in self.named_buffers():
            size = buffer.nelement() * buffer.element_size()  # number of elements * size of each element
            buffer_size += size
            buffers += buffer.nelement()

        total_size = param_size + buffer_size

        print("\nSummary:")
        print(f"  Number of parameters: {params / 1e6:.1f}M")
        print(f"  Total Model Size: {total_size / (1024 ** 2):.2f} MB")  # Convert to megabytes (MB)\

    def forward(self, x, target_seq=None):
        # Encode the input to get mu and logvar
        mu, logvar = self.encoder(x)  # Move x to the device
        # Reparameterization to get latent vector z
        z = self.reparameterize(mu, logvar)
        
        # Decode the latent vector z to generate a sequence
        logits_dict = self.decoder(z, target_seq=target_seq)
        
        return logits_dict, mu, logvar
    
    def generate(self, z, max_len=None, sos_token=1, eos_token=2):
        """
        Autoregressively generates a sequence using the decoder.

        Args:
            z: Latent vector (encoder output or random input), shape [batch_size, latent_dim].
            max_len: Maximum length of the generated sequence.

        Returns:
            generated_sequence: Generated sequence, shape [batch_size, generated_length].
        """
        max_len = max_len or self.max_len
        batch_size = z.size(0)
        device = z.device

        # Start with <sos> token for all batches
        generated_sequence = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

        # Decoder memory (latent representation) processing
        #memory = z.unsqueeze(1)  # Adjust shape if necessary for the decoder

        for step in range(max_len):
            # Generate the next token using the decoder
            output_logits = self.decoder(z, target_seq=generated_sequence)

            # Predict the next token (greedy decoding via argmax)
            next_token = torch.argmax(output_logits[:, -1, :], dim=-1, keepdim=True)  # Shape: [batch_size, 1]

            # Append the predicted token to the sequence
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

            # Stop early if all sequences generate <eos>
            if (next_token == eos_token).all():
                break

        return generated_sequence[:, :]

    
    def vae_loss(self, decoded_x, x, mu, logvar, kl_weight=1.0, free_bits=0.1, pad_token_idx=0):
        """VAE loss function for ADVAE combining over multiple attributes"""

        cross_entropies = ["mana", "name", "type", "oracle", "flavor", "power", "toughness", "loyalty"]
        # {
        #     "name_logits": name_logits,
        #     "mana_logits": mana_logits,
        #     "type_logits": type_logits,
        #     "oracle_logits": oracle_logits,
        #     "flavor_logits": flavor_logits,
        #     "power_logits": power,
        #     "toughness_logits": toughness,
        #     "loyalty_logits": loyalty,
        #     "has_pt": has_pt,
        #     "has_loyalty": has_loyalty
        # }

        total_loss = 0

        for attribute in cross_entropies:
            left = decoded_x[f"{attribute}_logits"].float()[:, :-1, :]  # Remove last prediction for target alignment
            right = x[f"{attribute}_tokens"][:, 1:].reshape(-1).to(device)
            recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token_idx)(left, right)
            total_loss += recon_loss

        
        
        
        def free_bits_kl(mu, logvar, free_bits=0.1):
            kl_loss_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            regularized_kl_loss = torch.clamp(kl_loss_per_dim, min=free_bits)
            return regularized_kl_loss.sum(dim=-1).mean()

        # KL divergence loss
        kld_loss = free_bits_kl(mu, logvar, free_bits=free_bits)

        scaled_kld_loss = kl_weight * kld_loss
        
        return recon_loss + scaled_kld_loss, recon_loss, kld_loss, scaled_kld_loss



