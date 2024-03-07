#References
#https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import csv2tokens
import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from tokenizers import Tokenizer

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
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
    
bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

if __name__ == "__main__":
    print(device)
    epochs = 20
    batch_size = 1


    data, tokenizer = csv2tokens.tokenize(file = 'cards.csv', features = ['name', 'mana_cost', 'type_line', 'power', 'toughness', 'oracle_text', 'flavor_text'])
    print(data.shape)
    #print(tokenizer.decode(data[5]))
    embed_dim = len(tokenizer)
    print(data)

    model = FblthpTransformerModel(ntoken=embed_dim, d_model=1000 * 2, nhead=5, d_hid=1000 * 4, nlayers= 4, dropout=0.2).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()
    data = batchify(data, 100)
    print(data)


    for epoch in range(epochs):
        epoch_loss = 0

        for i in range(0, data.size(0), bptt):
            batch, true = get_batch(data, i)
            optimizer.zero_grad()
            output = model.forward(batch)  
            output_flat = output.view(-1, embed_dim)
            loss = loss_func(output_flat, true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item() / batch.shape[0]

        print(f'{epoch}: {epoch_loss}')
    #using greedy choice for simplicity but we should do nucleus sampling instead later
    torch.save(torch.jit.script(model), 'model.pt')
    print('done Training')
    while True:
        command = input()
        if command == 'q':
            break
        current_string = command
        model.eval()
        for i in range(30):
            model_output = model.forward(torch.from_numpy(np.array(tokenizer(current_string)['input_ids'], dtype=np.int64)).to(device))[-1,-1,:]
            #print(model_output.shape)
            chosen = torch.argmax(model_output)
            out = tokenizer.decode(chosen)
            print(f"Output: {chosen}, decoded: '{out}'")
            current_string = current_string + " " + out
        print(current_string)




    


    
