import torch
import torch.nn as nn
import math

# Check if a GPU is available and use it if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class PoolingTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len=125):
        super(PoolingTransformerEncoder, self).__init__()
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len+2)
        
        # Transformer encoder layers with batch_first=True
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Linear layers for mean and log variance for reparameterization
        self.fc_mu = nn.Linear(embed_dim, embed_dim)  # Mean
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)  # Log variance
    
    def forward(self, x):
        # If x is 1D, add a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0) # Shape: [batch_size, sequence_length]
        
        # Embedding tokens into continuous space
        embedded = self.embedding(x)  # Shape: [batch_size, sequence_length, embed_dim]
        
        # Add positional encodings
        embedded = self.pos_encoder(embedded)
        
        # Pass through the transformer encoder with batch_first=True
        encoded = self.transformer_encoder(embedded)  # Shape: [batch_size, sequence_length, embed_dim]
        
        # Max pooling over the sequence length dimension
        pooled = torch.max(encoded, dim=1).values  # Shape: [batch_size, embed_dim]

        # # Average pooling over the sequence length dimension
        # pooled = torch.mean(encoded, dim=1)  # Shape: [batch_size, embed_dim]

        # --- Absolute pooling over the sequence length dimension ---
        # absolute_max_pooled = torch.max(encoded.abs(), dim=1).values

        # # Compute indices for the maximum absolute value in the sequence dimension
        # max_indices = encoded.abs().argmax(dim=1, keepdim=True)  # Shape: [batch_size, 1, embed_dim]

        # # Gather the values along the sequence dimension based on max_indices
        # max_sign_values = torch.gather(encoded, 1, max_indices).squeeze(1)  # Shape: [batch_size, embed_dim]

        # # Multiply absolute max pooled with the sign of these gathered max values
        # pooled = absolute_max_pooled * torch.sign(max_sign_values)

        ## --- End of absolute pooling ---

        # Get mean and log variance for reparameterization
        mu = self.fc_mu(pooled)  # Shape: [batch_size, embed_dim]
        logvar = self.fc_logvar(pooled)  # Shape: [batch_size, embed_dim]
        
        return mu, logvar

class PooledTransformerDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len=125, dropout_rate=0.1):
        super(PooledTransformerDecoder, self).__init__()
        self.output_dim = output_dim
        
        # Embedding layer for the output tokens
        self.embedding = nn.Embedding(output_dim, embed_dim)
        
        # Positional encoding for the decoder
        self.pos_encoder = PositionalEncoding(embed_dim, max_len + 2)
        
        # Transformer decoder layers with batch_first=True
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout_rate
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # Linear layer to map to vocabulary size
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout before the final layer
            nn.Linear(embed_dim, output_dim)
        )
        
        self.tgt_mask = torch.triu(torch.ones(max_len, max_len, device=device) * float('-inf'), diagonal=1)
        
        # Dropout for embeddings and positional encodings
        self.dropout = nn.Dropout(dropout_rate)
        
        # Store maximum length and special tokens
        self.max_len = max_len
        self.sos_token = 1
        self.eos_token = 2

    def forward(self, memory, target_seq=None, max_len=None, teacher_forcing_ratio=0.5):
        max_len = max_len or self.max_len
        batch_size = memory.size(0)
        device = memory.device
        
        # Initialize the output sequence with the <sos> token
        generated_sequence = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)
        BIG_OUTPUT_LOGIT_LIST = torch.empty((batch_size, max_len, self.output_dim), device=device)
        memory = memory.unsqueeze(1)
        
        for step in range(max_len):
            # Embed the current sequence
            embedded = self.embedding(generated_sequence)  # Shape: [batch_size, sequence_length, embed_dim]
            embedded = self.dropout(embedded)  # Dropout on embeddings
            embedded = self.pos_encoder(embedded)  # Add positional encodings
            embedded = self.dropout(embedded)  # Dropout after positional encodings
            
            # Pass through the transformer decoder
            current_mask = self.tgt_mask[:step + 1, :step + 1]
            decoded = self.transformer_decoder(tgt=embedded, memory=memory, tgt_mask=current_mask)
            
            # Predict the next token
            output_logits = self.fc_out(decoded[:, -1, :])  # Shape: [batch_size, output_dim]
            BIG_OUTPUT_LOGIT_LIST[:, step, :] = output_logits
            
            # Apply teacher forcing or auto-regressive generation
            temperature = 0.8
            scaled_output_logits = output_logits / temperature
            next_token = torch.argmax(scaled_output_logits, dim=-1)
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_token = target_seq[:, step]
            generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(1)], dim=1)
            
            # Stop early if all batches predict <eos>
            done = (next_token == self.eos_token)
            if done.all():
                break
        
        returned = generated_sequence[:, 1:]  # Remove <sos> token
        return returned, BIG_OUTPUT_LOGIT_LIST


class AverageAttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len=125, dropout_rate=0.1):
        super(AverageAttentionEncoder, self).__init__()
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len + 2)
        
        # Transformer encoder layers with batch_first=True
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, dropout=dropout_rate
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Dropout after the embedding layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Query and softmax for attention pooling
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.softmax = nn.Softmax(dim=-1)
        
        # Linear layers for mean and log variance
        self.fc_mu = nn.Linear(embed_dim, embed_dim)  # Mean
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)  # Log variance

    def forward(self, x):
        # If x is 1D, add a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Shape: [batch_size, sequence_length]
        
        # Embedding tokens into continuous space
        embedded = self.embedding(x)  # Shape: [batch_size, sequence_length, embed_dim]
        embedded = self.dropout(embedded)  # Apply dropout to embeddings
        
        # Add positional encodings
        embedded = self.pos_encoder(embedded)
        
        # Pass through the transformer encoder
        encoded = self.transformer_encoder(embedded)  # Shape: [batch_size, sequence_length, embed_dim]
        
        # Attention-based pooling
        attention_scores = torch.matmul(encoded, self.query)  # Shape: [batch_size, sequence_length]
        attention_weights = self.softmax(attention_scores)  # Shape: [batch_size, sequence_length]
        
        # Weighted sum of encoded representations
        attention_weights = attention_weights.unsqueeze(-1)  # Shape: [batch_size, sequence_length, 1]
        pooled = torch.sum(encoded * attention_weights, dim=1)  # Shape: [batch_size, embed_dim]
        
        # Compute mean and log variance for reparameterization
        mu = self.fc_mu(pooled)  # Shape: [batch_size, embed_dim]
        logvar = self.fc_logvar(pooled)  # Shape: [batch_size, embed_dim]
        
        return mu, logvar




class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len=125, dropout=0.1):
        super(TransformerVAE, self).__init__()
        
        # Encoder (Pooling Transformer Encoder)
        #self.encoder = PoolingTransformerEncoder(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len)
        self.encoder = AverageAttentionEncoder(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len, dropout)
        
        # Decoder (Pooled Transformer Decoder)
        self.decoder = PooledTransformerDecoder(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len, dropout)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random noise
        return mu + eps * std          # z = mu + eps * std

    def print_model_size(self, debug=False):
        param_size = 0
        params = 0

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

        print("\nBuffers:")
        for name, buffer in self.named_buffers():
            size = buffer.nelement() * buffer.element_size()  # number of elements * size of each element
            buffer_size += size
            buffers += buffer.nelement()
            print(f"  Name: {name}, Shape: {buffer.shape}, Memory: {size / (1024 ** 2):.2f} MB")

        total_size = param_size + buffer_size

        print("\nSummary:")
        print(f"  Number of parameters: {params}")
        print(f"  Number of buffers: {buffers}")
        print(f"  Total Model Size: {total_size / (1024 ** 2):.2f} MB")  # Convert to megabytes (MB)


    def forward(self, x, target_seq=None, max_len=None, teacher_forcing_ratio=0.5):
        # Encode the input to get mu and logvar
        mu, logvar = self.encoder(x.to(device))  # Move x to the device
        # Reparameterization to get latent vector z
        z = self.reparameterize(mu, logvar)
        
        # Decode the latent vector z to generate a sequence
        decoded_x, logits = self.decoder(z, target_seq=target_seq, max_len=max_len, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return decoded_x, logits, mu, logvar
    
    def generate(self, z, max_len=None):
        return self.decoder(z, target_seq=None, max_len=max_len)
    
    def vae_loss(self, decoded_x, x, mu, logvar, kl_weight=1.0, free_bits=0.1, pad_token_idx=0):
        """VAE loss function combining reconstruction loss and KL divergence with KL annealing"""
        decoded_x = decoded_x.float()
        left = decoded_x.view(-1, decoded_x.size(-1))
        right = x[:, 1:].reshape(-1).to(device)
        recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token_idx)(left, right)

        def free_bits_kl(mu, logvar, free_bits=0.1):
            kl_loss_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            regularized_kl_loss = torch.clamp(kl_loss_per_dim, min=free_bits)
            return regularized_kl_loss.sum(dim=-1).mean()


        
        # KL divergence loss
        kld_loss = free_bits_kl(mu, logvar, free_bits=free_bits)


        scaled_kld_loss = kl_weight * kld_loss
        
        return recon_loss + scaled_kld_loss, recon_loss, kld_loss, scaled_kld_loss



# if __name__ == "__main__":
#     # Define hyperparameters for the encoder
#     input_dim = 20000  # Example vocabulary size
#     embed_dim = 32  # Embedding size
#     num_heads = 8  # Number of attention heads in the transformer
#     hidden_dim = 1024  # Size of the feedforward layer in the transformer
#     num_layers = 2  # Number of transformer layers
#     max_len = 100  # Maximum length of sequences
    
#     # Create an instance of the PoolingTransformerEncoder
#     encoder = PoolingTransformerEncoder(input_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len).to(device)
    
#     # Generate a batch of random tokenized sequences as input (sequence_length, batch_size)
#     sequence_length = 50
#     batch_size = 2
#     input_tokens = torch.randint(0, input_dim, (sequence_length, batch_size)).to(device)  # Move to device
    
#     # Run the forward pass of the encoder
#     mu, logvar = encoder(input_tokens)
    
#     # Print out the results to verify shapes
#     print(mu)
#     print(logvar)
#     print(f"Mean (mu) shape: {mu.shape}")  # Should be [batch_size, embed_dim]
#     print(f"Log variance (logvar) shape: {logvar.shape}")  # Should be [batch_size, embed_dim]
