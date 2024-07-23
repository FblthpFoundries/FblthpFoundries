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
import random

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
        self.linear1 = nn.Linear(d_model, 2*d_model)
        self.linear2 = nn.Linear(2*d_model, (d_model+ntoken)//2)
        self.linear3 = nn.Linear((d_model+ntoken)//2, ntoken)
        self.dropout = nn.Dropout(dropout)
        self.sm = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.bias.data.zero_()
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear3.weight.data.uniform_(-initrange, initrange)

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
        output = self.linear3(self.dropout(self.linear2(self.dropout(self.linear1(output)))))
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

class CoolerPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len : int = 5000):
        
        super().__init__()
        section_list = ['<tl>', '<name>', '<mc>', '<ot>', '<power>', '<toughness>', '<ft>']
        token_idx = 0
        section_idx = 0
        embed = torch.nn.Embedding()

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
    epochs = 50
    batch_size = 100


    data, tokenizer = csv2tokens.tokenize(file = 'cards.csv', features = ['type_line', 'name', 'mana_cost', 'oracle_text', 'power', 'toughness',  'flavor_text'])
    tokenizer.save_pretrained('./models/tokenizer')
    print(data.shape)
    #print(tokenizer.decode(data[5]))
    embed_dim = len(tokenizer)
    print(data)

    

    model = FblthpTransformerModel(ntoken=embed_dim, d_model=100 * 2, nhead=5, d_hid=100 * 4, nlayers= 4, dropout=0.25).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()
    #data = batchify(data, batch_size)
    print(data.shape)
    data = data.to(device)

    
    for epoch in range(epochs):
        epoch_loss = 0
        start = random.randint(0,data.size(1) - batch_size)
        batch = data[:,start:start+batch_size]
        print(f'batch:{batch.shape}')

        output = model.forward(batch) #[seq, batch, ntokens]
        correct = torch.zeros(output.shape).to(device)
        for i in range(0, batch.size(0)):
            for j in range(0,batch.size(1)):
                correct[i,j,batch[i,j]] = 1

        loss= loss_func(output, correct)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item() / batch.shape[0]
        '''for i in range(1, batch.size(1)):
            optimizer.zero_grad()
            inp = batch[:,:i] #[batchsize, seq_len]
            output = model.forward(inp)
            correct = torch.zeros(output.shape).to(device)
            correct[batch[:,i]] = 1
            print('output')
            print(output.shape)
            print('correct')
            print(correct.shape)
            loss= loss_func(output, correct)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item() / batch.shape[0]

            batch, true = get_batch(data, i)
            optimizer.zero_grad()
            output = model.forward(batch)  
            output_flat = output.view(-1, embed_dim)
            loss = loss_func(output_flat, true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item() / batch.shape[0]'''

        print(f'{epoch}: {epoch_loss}')
    torch.save(model, './models/model.pt')
    #using greedy choice for simplicity but we should do nucleus sampling instead later
    print('done Training')




    


    
