import pandas
import argparse
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors, decoders 
import os
from transformers import GPT2TokenizerFast
import torch
import numpy as np
import re

#https://huggingface.co/learn/nlp-course/chapter6/8#building-a-bpe-tokenizer-from-scratch

specialTokenDict ={
    'type_line': '<tl>',
    'name': '<name>',
    'mana_cost': '<mc>',
    'oracle_text': '<ot>',
    'power': '<power>',
    'toughness': '<toughness>',
    'flavor_text': '<ft>',
    'eos' : '<|endoftext|>',
    'pad_token' : '<pad>',
    'nl': '<nl>'
}

pp = '\\+[0-9|X]+\\/\\+[0-9|X]+'
mm = '\\-[0-9|X]+\\/\\-[0-9|X]+'
xx = '[0-9|X]+\\/[0-9|X]+'
pm = '\\+[0-9|X]+\\/\\-[0-9|X]+'
mp = '\\-[0-9|X]+\\/\\+[0-9|X]+'


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
            info = str(row[feature])
            if not feature == 'name':
                info = info.replace(row['name'], '~')
            text += ' ' + specialTokenDict[feature] + ' ' + info + ' '
            
        #text += specialTokenDict['eos']

    data = torch.from_numpy(np.array(wrapped_tokeinizer(text)['input_ids'], dtype=np.int64))

    return data, wrapped_tokeinizer

def getCorpus(csv):
    df = pandas.read_csv(csv)

    corpus = []

    for index, row in df.iterrows():
        text= ''
        name = row['name']
        for feature in specialTokenDict:
            if feature not in row:
                continue
            append =' ' + specialTokenDict[feature] + ' ' + str(row[feature]) if not str(row[feature]) == '<empty>' else ''
            if not feature == 'name':
                append = append.replace(name, '~')
            text +=  append
        corpus.append(sanitize(text) + specialTokenDict['eos'])

    f = open('corpus.txt', 'w', encoding='utf-8')
    for text in corpus:
        f.write(text + '\n')
    f.close()
    return corpus

    
def sanitize(text):
    while '{' in text:
        indexOpen  = text.index('{')
        indexClose = text.index('}')

        symbol =text[indexOpen: indexClose + 1]
        token = '<' + text[indexOpen + 1: indexClose] + '>'

        if not symbol in specialTokenDict:
            specialTokenDict[symbol] = token

        text = text[: indexOpen] + token + text[indexClose + 1:]


    return text

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
