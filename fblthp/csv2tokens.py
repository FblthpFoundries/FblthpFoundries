import pandas
import argparse
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors, decoders 
import os
from transformers import GPT2TokenizerFast
import torch
import numpy as np
import re

#https://huggingface.co/learn/nlp-course/chapter6/8#building-a-bpe-tokenizer-from-scratch

featureDict ={
    'type_line': '<tl>',
    'name': '<name>',
    'mana_cost': '<mc>',
    'oracle_text': '<ot>',
    'power': '<power>',
    'toughness': '<toughness>',
    'loyalty' : '<loyalty>',
    'flavor_text': '<ft>',
    
}

specialTokenDict = {
    'type_line': '<tl>',
    'name': '<name>',
    'mana_cost': '<mc>',
    'oracle_text': '<ot>',
    'power': '<power>',
    'toughness': '<toughness>',
    'loyalty' : '<loyalty>',
    'flavor_text': '<ft>',
    'eos' : '<eos>',
    'pad_token' : '<pad>',
    'nl': '<nl>'
}

pp = '\\+[0-9|X]+/\\+[0-9|X]+'
mm = '\\-[0-9|X]+/\\-[0-9|X]+'
xx = '[0-9|X]+/[0-9|X]+'
pm = '\\+[0-9|X]+/\\-[0-9|X]+'
mp = '\\-[0-9|X]+/\\+[0-9|X]+'


def tokenize(file, features):
    if not os.path.isfile('tokenizer.json'):
        createTokenizer()
    tokenizer = Tokenizer.from_file('tokenizer.json')
    wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object = tokenizer)

    wrapped_tokenizer.add_special_tokens({'eos_token': '<eos>', 'pad_token':'<pad>'})

    csv = pandas.read_csv(file)
    csv = csv.sample(n=5000)#change to frac 1 for full set

    data = []
    npdata =  np.array([])
    

    for index, row in csv.iterrows():

        text = featureDict['eos']
        for feature in features:
            if feature == 'name':
                text+= ' ' + featureDict['name'] + ' ~ '
                continue
            info = str(row[feature])
            info = info.replace(row['name'], '~')
            text += ' ' + featureDict[feature] + ' ' + info + ' '
            
        text += featureDict['eos']

        data.append(np.array(wrapped_tokenizer(text)['input_ids'], dtype=np.int64))

    max_len = getMaxLen(data)
    for row in data:
        row = np.append(row, ([wrapped_tokenizer('<pad>')['input_ids']]*(max_len - len(row))))  
        if npdata.any():
            npdata = np.vstack((npdata, row))
        else:
            npdata = row


    data = torch.from_numpy(npdata).int().transpose(0,1)

    return data, wrapped_tokenizer

def getMaxLen(arr):
    maxLen = 0
    for row in arr:
        if len(row) > maxLen:
            maxLen=len(row)
    return maxLen

def getCorpus(csv):
    df = pandas.read_csv(csv)

    corpus = []

    for index, row in df.iterrows():
        text= ''
        name = row['name']
        for feature in featureDict:
            append = ' ' +featureDict[feature] 
            if feature in row:
                append += ' ' + str(row[feature]) if not str(row[feature]) == '<empty>' else '' 
            if not feature == 'name':
                append = append.replace(name, '~')
            text +=  append + ' ' + featureDict[feature][:1] + '\\' + featureDict[feature][1:]
        corpus.append(sanitize(text[1:]) + specialTokenDict['eos'])

    #corpus += getPlaneswalkers(df)

    f = open('corpus.csv', 'w', encoding='utf-8')
    f.write('card\n')
    for text in corpus:
        f.write('\"'+text + '\"\n')
    f.close()
    return corpus

def getPlaneswalkers(df):
    corpus = []

    for index, row in df.iterrows():
        if not 'Planeswalker' in row['type_line']:
            continue
        text= ''
        name = row['name']
        for feature in featureDict:
            append = ' ' +featureDict[feature] 
            if feature in row:
                append += ' ' + str(row[feature]) if not str(row[feature]) == '<empty>' else '' 
            if not feature == 'name':
                append = append.replace(name, '~')
            text +=  append + ' ' + featureDict[feature][:1] + '\\' + featureDict[feature][1:]
        corpus.append(sanitize(text[1:]) + specialTokenDict['eos'])
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

    special_tokens = [featureDict[key] for key in featureDict] 

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
