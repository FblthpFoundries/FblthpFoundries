import torch
import torch.nn as nn
import math

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, embed_dim) for positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute positional encodings using sine and cosine functions
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension (unsqueeze(0)) so it can be added to the input embeddings
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register pe as a buffer, meaning it won't be updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to the input embeddings
        x = x + self.pe[:x.size(0), :]
        return x

class PoolingTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len = 200):
        super(PoolingTransformerEncoder, self).__init__()
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Linear layers for mean and log variance for reparameterization
        self.fc_mu = nn.Linear(embed_dim, embed_dim)  # Mean
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)  # Log variance
    
    def forward(self, x):
        # If x is 1D, add a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0) # Shape: [batch_size, sequence_length]
        
        # Embedding tokens into continuous space
        #print("X Shape entering encoder: ", x.shape)
        embedded = self.embedding(x)  # Shape: [batch_size, sequence_length, embed_dim]
        
        # Add positional encodings
        embedded = self.pos_encoder(embedded)
        #print("Embedded Shape after Pos Encoding: ", embedded.shape)
        # Pass through the transformer encoder
        encoded = self.transformer_encoder(embedded)  # Shape: [batch_size, sequence_length, embed_dim]
        
        #print("Output from transformer encoder: ", encoded.shape)
        # Max pooling over the sequence length dimension
        pooled = torch.max(encoded, dim=1).values  # Shape: [batch_size, embed_dim]
        
        # Get mean and log variance for reparameterization
        mu = self.fc_mu(pooled)  # Shape: [batch_size, embed_dim]
        logvar = self.fc_logvar(pooled)  # Shape: [batch_size, embed_dim]
        
        return mu, logvar

# TransformerDecoder with dynamic sequence generation
class PooledTransformerDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len=200):
        super(PooledTransformerDecoder, self).__init__()
        self.output_dim = output_dim
        # Embedding layer for the output tokens
        self.embedding = nn.Embedding(output_dim, embed_dim)
        
        # Positional encoding for the decoder
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        
        # Transformer decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # Linear layer to map to vocabulary size
        self.fc_out = nn.Linear(embed_dim, output_dim)
        
        # Store maximum length and special tokens
        self.max_len = max_len
        self.sos_token = 0  # Assume <sos> token ID is 0
        self.eos_token = 1  # Assume <eos> token ID is 1
    
    def forward(self, memory, max_len=None):
        """
        memory: Latent vector from encoder, Shape: [batch_size, embed_dim]
        max_len: Maximum sequence length for generation (if not provided, use self.max_len)
        """
        # Set max_len if provided

        #print("Memory Shape (latent into Decoder): ", memory.shape)
        max_len = max_len or self.max_len
        
        batch_size = memory.size(0)
        device = memory.device
        
        # Initialize the output sequence with the <sos> token
        generated_sequence = torch.full((1, batch_size), self.sos_token, dtype=torch.long, device=device)

        #print("Generated Sequence Shape: ", generated_sequence.shape)
        
        # Expand memory (latent vector) to match the sequence length
        sequence_length = generated_sequence.size(0)  # Get current sequence length
        memory = memory.unsqueeze(0).expand(sequence_length, batch_size, -1)  # Shape: [sequence_length, batch_size, embed_dim]
        BIG_OUTUT_LOGIT_LIST = torch.zeros((0, batch_size, self.output_dim))
        #print("Memory Shape after expanding: ", memory.shape)
        
        for step in range(max_len):
            # Embed the current sequence
            embedded = self.embedding(generated_sequence)  # Shape: [sequence_length, batch_size, embed_dim]
            embedded = self.pos_encoder(embedded)  # Add positional encodings
            
            # Pass through the transformer decoder
            decoded = self.transformer_decoder(embedded, memory)  # Shape: [sequence_length, batch_size, embed_dim]
            
            # Predict the next token (only use the last token in the sequence)
            output_logits = self.fc_out(decoded[-1, :, :])  # Shape: [batch_size, output_dim]
            BIG_OUTUT_LOGIT_LIST = torch.cat([BIG_OUTUT_LOGIT_LIST, output_logits.unsqueeze(0)], dim=0)
            next_token = torch.argmax(output_logits, dim=-1)  # Shape: [batch_size]
            
            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(0)], dim=0)
            
            # If all batches predict <eos>, stop early
            if (next_token == self.eos_token).all():
                break
            #print("Generated Sequence Shape in loop: ", generated_sequence.shape)
        returned = generated_sequence[1:, :].transpose(0, 1)  # Remove <sos> token and transpose
        #print("Returned Shape: ", returned.shape)
        return returned, BIG_OUTUT_LOGIT_LIST# Return the generated sequence (excluding the initial <sos> token)

class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len=5000):
        super(TransformerVAE, self).__init__()
        
        # Encoder (Pooling Transformer Encoder)
        self.encoder = PoolingTransformerEncoder(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len)
        
        # Decoder (Pooled Transformer Decoder)
        self.decoder = PooledTransformerDecoder(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random noise
        return mu + eps * std          # z = mu + eps * std
    def print_model_size(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()  # number of elements * size of each element
        
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()  # number of elements * size of each element

        total_size = param_size + buffer_size
        print(f"Model Size: {total_size / (1024 ** 2):.2f} MB")  # Convert to megabytes (MB)
    def forward(self, x, max_len=None):
        # Encode the input to get mu and logvar
        mu, logvar = self.encoder(x)
        #print(f"Mu, Logvar shape: {mu.shape}, {logvar.shape}")
        # Reparameterization to get latent vector z
        z = self.reparameterize(mu, logvar)
        #print(f"Z shape: {z.shape}")
        
        # Decode the latent vector z to generate a sequence
        decoded_x, logits = self.decoder(z, max_len)
        
        return decoded_x, logits, mu, logvar
    
    def generate(self, z, max_len=None):
        return self.decoder(z, max_len)
    
    def vae_loss(self, decoded_x, x, mu, logvar):
        """VAE loss function combining reconstruction loss and KL divergence"""
        # Reconstruction loss (cross-entropy between true tokens and predicted tokens)
        decoded_x = decoded_x.float()
        left = decoded_x.view(-1, decoded_x.size(-1))
        right = x.view(-1)
        recon_loss = nn.CrossEntropyLoss()(left, right)
        
        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kld_loss


if __name__ == "__main__":
    # Define hyperparameters for the encoder
    input_dim = 10000  # Example vocabulary size
    embed_dim = 512  # Embedding size
    num_heads = 8  # Number of attention heads in the transformer
    hidden_dim = 1024  # Size of the feedforward layer in the transformer
    num_layers = 6  # Number of transformer layers
    max_len = 100  # Maximum length of sequences
    
    # Create an instance of the PoolingTransformerEncoder
    encoder = PoolingTransformerEncoder(input_dim, embed_dim, num_heads, hidden_dim, num_layers, max_len)
    
    # Generate a batch of random tokenized sequences as input (sequence_length, batch_size)
    sequence_length = 50
    batch_size = 4
    input_tokens = torch.randint(0, input_dim, (sequence_length, batch_size))  # Random integer tokens
    
    # Run the forward pass of the encoder
    mu, logvar = encoder(input_tokens)
    # Print out the results to verify shapes
    print(mu)
    print(logvar)
    print(f"Mean (mu) shape: {mu.shape}")  # Should be [batch_size, embed_dim]
    print(f"Log variance (logvar) shape: {logvar.shape}")  # Should be [batch_size, embed_dim]
