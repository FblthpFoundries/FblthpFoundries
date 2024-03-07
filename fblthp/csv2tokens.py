import pandas
import argparse
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors, decoders 
import os
from transformers import GPT2TokenizerFast
import torch
import numpy as np

#https://huggingface.co/learn/nlp-course/chapter6/8#building-a-bpe-tokenizer-from-scratch

specialTokenDict ={
    'name': '<name>',
    'mana_cost': '<mc>',
    'type_line': '<tl>',
    'power': '<power>',
    'toughness': '<toughness>',
    'oracle_text': '<ot>',
    'flavor_text': '<ft>',
    'eos' : '<|endoftext|>',
    'pad_token' : '<pad>'
}

def tokenize(file, features):
    if not os.path.isfile('tokenizer.json'):
        createTokenizer()
    tokenizer = Tokenizer.from_file('tokenizer.json')
    wrapped_tokeinizer = GPT2TokenizerFast(tokenizer_object = tokenizer)

    wrapped_tokeinizer.add_special_tokens({'eos_token': '<|endoftext|>', 'pad_token':'<pad>'})

    text = ''
    csv = pandas.read_csv(file)

    for index, row in csv.iterrows():

        #if index > 1000:
        #    break
        text += specialTokenDict['eos']
        for feature in features:
            text += ' ' + specialTokenDict[feature] + ' ' + str(row[feature]) + ' '
        text += specialTokenDict['eos']

    data = torch.from_numpy(np.array(wrapped_tokeinizer(text)['input_ids'], dtype=np.int64))

    return data, wrapped_tokeinizer

def getCorpus(csv):
    df = pandas.read_csv(csv)

    corpus = []

    for index, row in df.iterrows():
        text= ''
        for feature in specialTokenDict:
            if feature not in row:
                continue
            text +=   ' ' + str(row[feature]) if not str(row[feature]) == '<empty>' else ''
        corpus.append(text)

    f = open('corpus.txt', 'w', encoding='utf-8')
    for text in corpus:
        f.write(text + '\n')
    f.close()
    return corpus

    

def createTokenizer(csv = 'cards.csv', k = 30_000):

    corpus = getCorpus(csv)
    special_tokens = [specialTokenDict[key] for key in specialTokenDict] 

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.post_processors = processors.ByteLevel(trim_offsets = True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(vocab_size = k, special_tokens = special_tokens)
    tokenizer.train(['corpus.txt'], trainer=trainer)
    tokenizer.save('tokenizer.json')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Loads CSV of card data and builds BPE tokenizer')
    parse.add_argument('-f', '--file', help='File path holding CSV of card data', type=str,  required=False)
    parse.add_argument('-k', help='Hyperparameter of vocabulary size for BPE', type=int, required=False)
    args = parse.parse_args()

    csv = args.file if args.file else 'cards.csv'
    k = args.k if args.k else 30_000

    createTokenizer(csv=csv, k=k)
