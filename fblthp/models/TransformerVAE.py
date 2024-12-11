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


class BertCLSEncoder(nn.Module):
    def __init__(self, bert_model_path, latent_dim, dropout_rate=0.1):
        super(BertCLSEncoder, self).__init__()
        
        # Load a fine-tuned BERT model
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Linear layers to compute mean and log variance for latent vector
        hidden_dim = self.bert.config.hidden_size  # Hidden size of BERT's encoder
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): Tokenized input IDs, shape [batch_size, sequence_length].
            attention_mask (torch.Tensor): Attention mask, shape [batch_size, sequence_length].
        
        Returns:
            mu (torch.Tensor): Latent mean, shape [batch_size, latent_dim].
            logvar (torch.Tensor): Latent log variance, shape [batch_size, latent_dim].
        """
        # Get outputs from the fine-tuned BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the CLS token's hidden state (first token in the sequence)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
        
        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)
        
        # Compute mean and log variance for the latent space
        mu = self.fc_mu(cls_embedding)        # Shape: [batch_size, latent_dim]
        logvar = self.fc_logvar(cls_embedding)  # Shape: [batch_size, latent_dim]
        
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

    def forward(self, memory, target_seq):
        """
        Forward pass with 100% teacher forcing.
        
        Args:
            memory: Encoder outputs, shape [batch_size, memory_dim].
            target_seq: Ground truth target sequence, shape [batch_size, seq_len].
        
        Returns:
            output_logits: Predicted logits for all steps, shape [batch_size, seq_len, output_dim].
        """
        device = memory.device
        batch_size, seq_len = target_seq.size()

        # Embed the entire target sequence (assumes <sos> is already prepended to target_seq)
        embedded = self.embedding(target_seq)  # Shape: [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)  # Dropout on embeddings
        embedded = self.pos_encoder(embedded)  # Add positional encodings
        embedded = self.dropout(embedded)  # Dropout after positional encodings
        # Prepare the causal mask for the target sequence
        current_mask = self.tgt_mask[:seq_len, :seq_len]

        # Pass through the transformer decoder
        decoded = self.transformer_decoder(
            tgt=embedded, 
            memory=memory.unsqueeze(1), 
            tgt_mask=current_mask
        )  # Shape: [batch_size, seq_len, embed_dim]

        # Compute output logits for all time steps
        output_logits = self.fc_out(decoded)  # Shape: [batch_size, seq_len, output_dim]


        return output_logits


class GPT2Decoder(nn.Module):
    def __init__(self, gpt2_path, latent_dim, output_dim, max_len=125, dropout_rate=0.1):
        super(GPT2Decoder, self).__init__()

        # Load GPT-2 configuration and model
        self.gpt2_config = GPT2Config.from_pretrained(gpt2_path)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_path)

        # Project latent vector to GPT-2 hidden size
        self.latent_to_gpt = nn.Linear(latent_dim, self.gpt2_config.n_embd)

        # Parameters
        self.output_dim = output_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout_rate)
        self.sos_token = 1
        self.eos_token = 2

    def forward(self, z, target_seq):
        """
        Forward pass for the GPT2Decoder with 100% teacher forcing.

        Args:
            z (torch.Tensor): Latent vector, shape [batch_size, latent_dim].
            target_seq (torch.Tensor): Ground truth target sequence, shape [batch_size, sequence_length].
            max_len (int): Maximum sequence length (optional).
            teacher_forcing_ratio (float): Teacher forcing ratio (must be 1.0 for this implementation).

        Returns:
            logits (torch.Tensor): Output logits, shape [batch_size, sequence_length, output_dim].
        """
        assert target_seq is not None, "Target sequence must be provided for teacher forcing."

        batch_size, seq_len = target_seq.size()
        max_len = max_len or self.max_len

        # Ensure the latent vector is projected to the GPT-2 hidden state size
        initial_hidden_state = self.latent_to_gpt(z).unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]

        # Prepare input embeddings for GPT-2
        gpt2_input_embeds = self.gpt2.transformer.wte(target_seq)  # Shape: [batch_size, seq_len, hidden_size]

        # Add initial latent state as the first token embedding
        gpt2_input_embeds = torch.cat([initial_hidden_state, gpt2_input_embeds[:, :-1, :]], dim=1)

        # Pass through GPT-2 model
        gpt2_outputs = self.gpt2(inputs_embeds=gpt2_input_embeds)

        # Output logits for vocabulary prediction
        logits = gpt2_outputs.logits  # Shape: [batch_size, seq_len, output_dim]

        return logits


class BertVAE(nn.Module):
    def __init__(self, bert_model_path, gpt2_path, latent_dim, vocab_size, max_len=125, dropout_rate=0.1):
        super(BertVAE, self).__init__()
        
        # Encoder (BERT with CLS token)
        self.encoder = BertCLSEncoder(bert_model_path, latent_dim, dropout_rate)
        
        # Decoder (GPT2Decoder)
        self.decoder = GPT2Decoder(gpt2_path, latent_dim, vocab_size, max_len, dropout_rate)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random noise
        return mu + eps * std          # z = mu + eps * std

    def forward(self, input_ids, attention_mask, target_seq=None, max_len=None):
        """
        Forward pass through the VAE.
        
        Args:
            input_ids (torch.Tensor): Input token IDs, shape [batch_size, sequence_length].
            attention_mask (torch.Tensor): Attention mask for BERT, shape [batch_size, sequence_length].
            target_seq (torch.Tensor): Target sequence for decoder, shape [batch_size, target_length].
            max_len (int): Maximum length for generation (optional).
        
        Returns:
            decoded_x (torch.Tensor): Generated sequences, shape [batch_size, max_len].
            logits (torch.Tensor): Logits for each token in the output, shape [batch_size, max_len, vocab_size].
            mu (torch.Tensor): Latent mean, shape [batch_size, latent_dim].
            logvar (torch.Tensor): Latent log variance, shape [batch_size, latent_dim].
        """
        # Encode the input to get mu and logvar
        mu, logvar = self.encoder(input_ids, attention_mask)
        
        # Reparameterization trick to get latent vector z
        z = self.reparameterize(mu, logvar)
        
        # Decode the latent vector to generate sequences
        logits = self.decoder(z, target_seq=target_seq, max_len=max_len)
        
        return logits, mu, logvar

    def generate(self, z, max_len=None, temperature=1.0):
        """
        Generate sequences from the latent vector.

        Args:
            z (torch.Tensor): Latent vector, shape [batch_size, latent_dim].
            max_len (int): Maximum length of the sequence to generate.
            temperature (float): Temperature parameter for sampling; lower values make the model more confident, higher values make it more random.

        Returns:
            torch.Tensor: Generated sequences, shape [batch_size, max_len].
        """
        max_len = max_len or self.decoder.max_len  # Use the decoder's default max length if none is provided
        batch_size = z.size(0)
        device = z.device
        
        # Initialize the sequence with the <sos> token
        generated_sequence = torch.full((batch_size, 1), self.decoder.sos_token, dtype=torch.long, device=device)
        
        for step in range(max_len):
            # Prepare the input embeddings for the decoder
            embedded = self.decoder.gpt2.transformer.wte(generated_sequence)  # Embed the current sequence
            latent_hidden_state = self.decoder.latent_to_gpt(z).unsqueeze(1)  # Project latent vector to GPT-2 hidden size
            
            # Concatenate latent state with current embeddings
            gpt2_input_embeds = torch.cat([latent_hidden_state, embedded[:, :-1, :]], dim=1)
            
            # Pass through GPT-2 to get logits
            gpt2_outputs = self.decoder.gpt2(inputs_embeds=gpt2_input_embeds)
            logits = gpt2_outputs.logits[:, -1, :]  # Take logits of the last generated token
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Sample next token or choose argmax
            next_token = torch.argmax(scaled_logits, dim=-1)
            
            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(1)], dim=1)
            
            # Stop generation if all sequences have predicted <eos>
            if (next_token == self.decoder.eos_token).all():
                break
        
        return generated_sequence[:, 1:]  # Remove <sos> token from the output

    def vae_loss(self, decoded_x, target_seq, mu, logvar, kl_weight=1.0, free_bits=0.1, pad_token_idx=0):
        """
        VAE loss function combining reconstruction loss and KL divergence.
        
        Args:
            decoded_x (torch.Tensor): Output logits from the decoder, shape [batch_size, seq_len, vocab_size].
            target_seq (torch.Tensor): Ground truth target sequence, shape [batch_size, seq_len].
            mu (torch.Tensor): Latent mean, shape [batch_size, latent_dim].
            logvar (torch.Tensor): Latent log variance, shape [batch_size, latent_dim].
            kl_weight (float): Weight for the KL divergence loss.
            free_bits (float): Minimum allowed KL divergence per dimension.
            pad_token_idx (int): Padding token index to ignore in the loss.
        
        Returns:
            tuple: Total loss, reconstruction loss, KL divergence, scaled KL divergence.
        """
        # Reconstruction loss


        decoded_x = decoded_x.float()
        left = decoded_x.view(-1, decoded_x.size(-1))  # Flatten for CrossEntropyLoss
        right = target_seq[:, 1:].reshape(-1).to(device)  # Shift target to align with predictions
        print(left, right)
        recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token_idx)(left, right)

        # KL divergence with free bits
        def free_bits_kl(mu, logvar, free_bits=0.1):
            kl_loss_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            regularized_kl_loss = torch.clamp(kl_loss_per_dim, min=free_bits)
            return regularized_kl_loss.sum(dim=-1).mean()

        kld_loss = free_bits_kl(mu, logvar, free_bits=free_bits)

        # Scale KL divergence
        scaled_kld_loss = kl_weight * kld_loss
        
        # Total loss
        return recon_loss + scaled_kld_loss, recon_loss, kld_loss, scaled_kld_loss

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
        print(f"  Total Model Size: {total_size / (1024 ** 2):.2f} MB")  # Convert to megabytes (MB)\
class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len=125, dropout=0.1):
        super(TransformerVAE, self).__init__()
        
        # Encoder (Pooling Transformer Encoder)
        #self.encoder = PoolingTransformerEncoder(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len)
        self.encoder = AverageAttentionEncoder(vocab_size, embed_dim, num_heads, hidden_dim, num_encoder_layers, max_len, dropout)
        
        # Decoder (Pooled Transformer Decoder)
        self.decoder = PooledTransformerDecoder(vocab_size, embed_dim, num_heads, hidden_dim, num_decoder_layers, max_len, dropout)

    
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
        if debug:
            print("\nBuffers:")
        for name, buffer in self.named_buffers():
            size = buffer.nelement() * buffer.element_size()  # number of elements * size of each element
            buffer_size += size
            buffers += buffer.nelement()
            if debug:
                print(f"  Name: {name}, Shape: {buffer.shape}, Memory: {size / (1024 ** 2):.2f} MB")

        total_size = param_size + buffer_size

        print("\nSummary:")
        print(f"  Number of parameters: {params}")
        print(f"  Number of buffers: {buffers}")
        print(f"  Total Model Size: {total_size / (1024 ** 2):.2f} MB")  # Convert to megabytes (MB)\

    def forward(self, x, target_seq=None, max_len=None):
        # Encode the input to get mu and logvar
        mu, logvar = self.encoder(x.to(device))  # Move x to the device
        # Reparameterization to get latent vector z
        z = self.reparameterize(mu, logvar)
        
        # Decode the latent vector z to generate a sequence
        logits = self.decoder(z, target_seq=target_seq)
        
        return logits, mu, logvar
    
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

        # Remove the initial <sos> token
        return generated_sequence[:, 1:]

    
    def vae_loss(self, decoded_x, x, mu, logvar, kl_weight=1.0, free_bits=0.1, pad_token_idx=0):
        """VAE loss function combining reconstruction loss and KL divergence with KL annealing"""
        decoded_x = decoded_x.float()[:, :-1, :]  # Remove last prediction for target alignment
        left = decoded_x.contiguous().view(-1, decoded_x.size(-1))
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
