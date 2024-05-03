import argparse
import torch
from tokenizers import Tokenizer
from torch import nn, Tensor, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from language_model import FblthpTransformerModel, PositionalEncoding
from transformers import AutoTokenizer
import numpy as np
import random

def main(args):
    model = args.model
    number = 1 if not args.number else args.number
    temp = 0 if not args.temp else args.temp
    out = args.out
    verbose = args.verbose
    if verbose:
        print(args)

    gen = torch.load(model)
    gen.eval()

    tokenizer = AutoTokenizer.from_pretrained('./models/tokenizer/')
    cards = []

    for i in range(number):
        cards.append(genCard(gen, tokenizer, temp, verbose))

    with open(out, 'w', encoding='utf8') as f:
        for card in cards:
            f.write(card + '\n')


def genCard(model, tokenizer, temp, verbose):
    sm = torch.nn.Softmax(dim=0)
    iterations = 0
    currentString = '<tl>'
    pred = ''

    while not pred == '<|endoftext|>' and iterations < 50 :
        if verbose:
            print(f'current string: {currentString}')
        model_output = model.forward(torch.from_numpy(np.array(tokenizer(currentString)['input_ids'], dtype=np.int64)).to(device))[-1,-1,:]
        indices = sample(sm(model_output), temp)

        if verbose:
            string = 'Sampled words:'
            for index in indices:
                prob = sm(model_output)[index]
                string += f' {tokenizer.decode(index)}({prob}),'
            string = string[:-1]
            print(string)
        chosen = random.choice(indices)

        pred = tokenizer.decode(chosen)

        if verbose:
            print(f'Chose: {pred}')

        currentString += pred
        iterations += 1

    return currentString


def sample(model_output, temp):
    index = torch.argmax(model_output).item()
    prob = model_output[index].item()
    model_output[index] = 0
    indices = [index]

    while prob < temp:
        index = torch.argmax(model_output).item()
        prob += model_output[index].item()
        model_output[index] = 0
        indices.append(index)
    return indices



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--model','-m', type=str, required=True, help='pt file of model')
    parse.add_argument('--number','-n', type=int, required=False, default=1, help='number of cards to generate')
    parse.add_argument('--temp', '-t', type=float, required=False, default= 0, help='Tempurature for nucleus sampling')
    parse.add_argument('--out', '-o', type=str, required=False, default='out.txt', help='file location for output')
    parse.add_argument('--verbose', '-v', required=False, default=False, help='Prints addition information as generating')
    args = parse.parse_args()

    main(args)
