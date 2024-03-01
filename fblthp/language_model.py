#References
#https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import csv2tokens

import torch
from torch import nn, Tensor, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from tokenizers import Tokenizer

class FblthpTransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.sm = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))#.to(device)
        output = self.transformer_encoder(src)#, src_mask)
        output = self.linear(output)
        return self.sm(output)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len : int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        """
                Arguments:
                    x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    epochs = 5
    batch_size = 1


    data, tokenizer = csv2tokens.tokenize(file = 'cards.csv', features = ['name', 'mana_cost', 'type_line', 'power', 'toughness', 'oracle_text', 'flavor_text'])
    print(data.shape)
    
    embed_dim = len(tokenizer)

    model = FblthpTransformerModel(ntoken=embed_dim, d_model=1000 * 2, nhead=5, d_hid=1000 * 4, nlayers= 4)
    model.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.NLLLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        indices = torch.randperm(data.shape[0])
        data = data[indices]
        batches = [data[i:i+batch_size] for i in range(0, data.shape[0], batch_size)]

        
        print(batches[0].shape)
        optimizer.zero_grad()
        true = torch.zeros((batches[0].shape[0], embed_dim))
        output = model.forward(batches[0])
        loss = loss_func(output, true)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item / batches[0].shape[0]

        print(f'{epoch}: {epoch_loss}')

    for i in range(30):
        out = model.forward(tokenizer('<|endoftext|>')['input_ids'])
        print(tokenizer.decode(out))



    


    