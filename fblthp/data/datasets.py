import os
import pandas as pd
import requests
from tokenizers import Tokenizer
from .mana_vocab import ManaVocabulary
import re
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split
import ast

DATA_DIR = os.path.dirname(os.path.abspath(__file__))



class MagicCardDataset(Dataset):
    def __init__(self, max_len=125, target="labeled.csv", prepare_corpus=False, tokenizer_path=os.path.join(DATA_DIR, "wordpiece_tokenizer.json")):
        self.tokenizer_path = tokenizer_path
        self.max_len = max_len
        
        if prepare_corpus:
            self.prepare_corpus()

        self.corpus_dataframe = pd.read_csv(target)

    def __len__(self):
        return len(self.corpus_dataframe)

    def __getitem__(self, idx):
        row = self.corpus_dataframe.iloc[idx]
        return row.tolist()
    
    def prepare_corpus(self):
        if not os.path.exists(os.path.join(DATA_DIR, 'cards.csv')):
            self.pull_scryfall_to_csv(os.path.join(DATA_DIR, 'cards.csv'))
        if not os.path.exists(os.path.join(DATA_DIR, 'labeled.csv')):
            self._prepare_corpus(os.path.join(DATA_DIR, 'cards.csv'), os.path.join(DATA_DIR, 'labeled.csv'))
    @staticmethod
    def pull_scryfall_to_csv(filename):
        print("Pulling data from Scryfall API...")
        response = requests.get('https://api.scryfall.com/bulk-data/oracle-cards')
        response.raise_for_status()
        
        json_uri = response.json()['download_uri']
        card_data = requests.get(json_uri).json()

        features = ['mana_cost', 'name', 'type_line', 'power', 'toughness', 'oracle_text', 'loyalty', 'flavor_text']
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(','.join(features) + '\n')
            for card in card_data:
                if 'Token' in card['type_line'] or 'card_faces' in card or 'paper' not in card['games']:
                    continue
                row = []
                for feature in features:
                    thing = card.get(feature, "<empty>").replace("\"", "").replace("\n", " <nl> ").replace("}{", "} {" )
                    row.append(f'"{thing}"')
                f.write(','.join(row) + '\n')

        print(f"Data saved to {filename}")

    def _prepare_corpus(self, cards_csv, filename):
        print("Preparing corpus...")
        df = pd.read_csv(cards_csv)

        feature_dict = {
            'type_line': '<tl>',
            'name': '<name>',
            'mana_cost': '<mc>',
            'oracle_text': '<ot>',
            'power': '<power>',
            'toughness': '<toughness>',
            'loyalty': '<loyalty>',
            'flavor_text': '<ft>',
        }

        special_tokens = {
            'eos': '<eos>'
        }

        def yoink(text, attribute):
            pattern = fr"<{attribute}>(.*?)<\\{attribute}>"
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
            return None
        text_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        o_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        o_tokenizer.enable_padding(pad_id=0, pad_token='<pad>', length=90)
        o_tokenizer.enable_truncation(max_length=90)
        n_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        n_tokenizer.enable_padding(pad_id=0, pad_token='<pad>', length=20)
        n_tokenizer.enable_truncation(max_length=20)
        t_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        t_tokenizer.enable_padding(pad_id=0, pad_token='<pad>', length=15)
        t_tokenizer.enable_truncation(max_length=15)
        f_tokenizer = Tokenizer.from_file(self.tokenizer_path)
        f_tokenizer.enable_padding(pad_id=0, pad_token='<pad>', length=50)
        f_tokenizer.enable_truncation(max_length=50)

        mana_tokenizer = ManaVocabulary()
        corpus = pd.DataFrame(columns=['card'])
        for _, row in df.iterrows():
            text = ''
            for feature, token in feature_dict.items():
                value = row.get(feature, '<empty>')
                text += f' {token} {value} {token[:1]}\\{token[1:]}'
            card = text.strip() + f' {special_tokens["eos"]}'
            #print(corpus)
            #print(card)
            corpus.loc[len(corpus)] = [card]
        
        corpus["tokens"] = corpus["card"].apply(lambda x: text_tokenizer.encode(x, ).ids)
        corpus["mc"] = corpus["card"].apply(lambda x: yoink(x, "mc"))
        corpus["power"] = corpus["card"].apply(lambda x: yoink(x, "power"))
        corpus["toughness"] = corpus["card"].apply(lambda x: yoink(x, "toughness"))
        corpus["cmc"] = corpus["mc"].apply(lambda x: sum([int(y) if y.isdigit() else 1 for y in x.replace(" ", "").replace("{", "").replace("}", "")]))
        corpus["power"] = corpus["power"].apply(lambda x: int(x) if x.isdigit() else x)
        corpus["toughness"] = corpus["toughness"].apply(lambda x: int(x) if x.isdigit() else x)

        corpus["oracle_text"] = corpus["card"].apply(lambda x: yoink(x, "ot"))
        corpus["name"] = corpus["card"].apply(lambda x: yoink(x, "name"))
        corpus["type_line"] = corpus["card"].apply(lambda x: yoink(x, "tl"))
        corpus["flavor_text"] = corpus["card"].apply(lambda x: yoink(x, "ft"))

        corpus["oracle_tokens"] = corpus["oracle_text"].apply(
            lambda x: o_tokenizer.encode(x).ids)
        corpus["name_tokens"] = corpus["name"].apply(
            lambda x: n_tokenizer.encode(x).ids)
        corpus["type_line_tokens"] = corpus["type_line"].apply(
            lambda x: t_tokenizer.encode(x).ids)
        corpus["flavor_text_tokens"] = corpus["flavor_text"].apply(
            lambda x: f_tokenizer.encode(x).ids)
        corpus["mana_tokens"] = corpus["mc"].apply(lambda x: mana_tokenizer.encode(x, pad_to_length=18).tolist())
            

        corpus.to_csv(f'{filename}', index=False)

        print(f"Corpus saved to {filename}")


def collate_fn(batch):
        def parse(text):
            parsed_list = ast.literal_eval(text)
            if parsed_list[0] != 1:
                parsed_list = [1] + parsed_list[:-1]
            return parsed_list
        
        #print(batch)
        ids = [parse(x[1]) for x in batch]
        ids = torch.tensor(ids)
        originals = [x[0] for x in batch]
        mc = [x[2] for x in batch]
        power = [x[3] for x in batch]
        toughness = [x[4] for x in batch]
        cmc = [x[5] for x in batch]
        ot = [x[6] for x in batch]
        name = [x[7] for x in batch]
        tl = [x[8] for x in batch]
        ft = [x[9] for x in batch]
        ot_tok = [parse(x[10]) for x in batch]
        name_tok = [parse(x[11]) for x in batch]
        tl_tok = [parse(x[12]) for x in batch]
        ft_tok = [parse(x[13]) for x in batch]
        mana_tok = [parse(x[14]) for x in batch]

        return {
            "ids": ids, 
            "originals": originals,
            "mc": mc,
            "power": power,
            "toughness": toughness,
            "cmc": cmc,
            "oracle_text": ot,
            "name": name,
            "type_line": tl,
            "flavor_text": ft,
            "oracle_tokens": ot_tok,
            "name_tokens": name_tok,
            "type_line_tokens": tl_tok,
            "flavor_text_tokens": ft_tok,
            "mana_tokens": mana_tok
            }

def get_dataloaders(test_set_portion, seed, batch_size):
    # DataLoader
    dataset = MagicCardDataset(target="data/labeled.csv", prepare_corpus=True)

    # Split the dataset into training and validation sets
    train_size = int(len(dataset) * (1 - test_set_portion))
    test_size = len(dataset) - train_size
    torch.manual_seed(seed) # Seed so the random split is the same and doesn't contaminate when reloading a model
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                 pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    td, tsd = get_dataloaders(0, 42, 1)
    datas = [[],[],[],[],[]]
    maxes = [0,0,0,0,0]
    max_items = [None, None, None, None, None]
    for batch in td:
        mana_seq_len = torch.nonzero(torch.tensor(batch["mana_tokens"])).size(0)
        datas[0].append(mana_seq_len)
        if mana_seq_len > maxes[0]:
            maxes[0] = mana_seq_len
            max_items[0] = torch.tensor(batch["mana_tokens"])
        
        
        oracle_seq_len = torch.nonzero(torch.tensor(batch["oracle_tokens"])).size(0)
        datas[1].append(oracle_seq_len)
        if oracle_seq_len > maxes[1]:
            maxes[1] = oracle_seq_len
            max_items[1] = torch.tensor(batch["oracle_tokens"])

        name_seq_len = torch.nonzero(torch.tensor(batch["name_tokens"])).size(0)
        datas[2].append(name_seq_len)
        if name_seq_len > maxes[2]:
            maxes[2] = name_seq_len
            max_items[2] = torch.tensor(batch["name_tokens"])

        type_line_seq_len = torch.nonzero(torch.tensor(batch["type_line_tokens"])).size(0)
        datas[3].append(type_line_seq_len)
        if type_line_seq_len > maxes[3]:
            maxes[3] = type_line_seq_len
            max_items[3] = torch.tensor(batch["type_line_tokens"])
        
        flavor_text_seq_len = torch.nonzero(torch.tensor(batch["flavor_text_tokens"])).size(0)
        datas[4].append(flavor_text_seq_len)
        if flavor_text_seq_len > maxes[4]:
            maxes[4] = flavor_text_seq_len
            max_items[4] = torch.tensor(batch["flavor_text_tokens"])
    print(maxes)

    import numpy as np
    import matplotlib.pyplot as plt

    tokenizer = Tokenizer.from_file(os.path.join(DATA_DIR, "wordpiece_tokenizer.json"))
    # for item in max_items [1:]:
    #     print(item)
    #     print(tokenizer.decode(item.tolist()[0]))

    for array in datas:
        data = np.array(array)
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins='auto', edgecolor='black')
        plt.title('Histogram of Text Sequence Lengths')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.show()

    #17 90 20 15 50
    
