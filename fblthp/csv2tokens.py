import pandas
import argparse
from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors, decoders 

#https://huggingface.co/learn/nlp-course/chapter6/8#building-a-bpe-tokenizer-from-scratch

specialTokenDict ={
    'name': '<name>',
    'mana_cost': '<mc>',
    'type_line': '<tl>',
    'power': '<power>',
    'toughness': '<toughness>',
    'oracle_text': '<ot>',
    'flavor_text': '<ft>'
}

def getCorpus(csv):
    df = pandas.read_csv(csv)

    corpus = []

    for index, row in df.iterrows():
        text= ''
        for feature in specialTokenDict:
            text +=   ' ' + str(row[feature]) if not str(row[feature]) == '<empty>' else ''
        corpus.append(text)

    f = open('corpus.txt', 'w', encoding='utf-8')
    for text in corpus:
        f.write(text + '\n')
    f.close()
    return corpus

    

def tokenize(csv = 'cards.csv', k = 30_000):

    corpus = getCorpus(csv)
    special_tokens = [specialTokenDict[key] for key in specialTokenDict]
    special_tokens.append('<|endoftext|>')

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

    tokenize(csv=csv, k=k)
