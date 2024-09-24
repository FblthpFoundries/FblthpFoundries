import torch
import torch.nn as nn
import torch.optim as optim

class UrzasTransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(UrzasTransformerEncoderDecoder, self).__init__()

        self.model_type = 'Transformer'
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer Encoder-Decoder
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )

        # Output Layer (linear projection to vocab size)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask):
        # Embedding + Positional Encoding for input and target
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)

        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt = self.pos_encoder(tgt)

        # Pass through transformer
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)

        # Project the final output to vocabulary size
        return self.fc_out(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)