import os
import pandas as pd
import requests
from tokenizers import Tokenizer
import re
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split
import ast

class MagicCardDataset(Dataset):
    def __init__(self, max_len=125, target="labeled.csv", prepare_corpus=False, tokenizer_path="wordpiece_tokenizer.json"):
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
        if not os.path.exists('cards.csv'):
            self.pull_scryfall_to_csv('cards.csv')
        if not os.path.exists('labeled.csv'):
            self.prepare_corpus('cards.csv', 'labeled.csv')
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

    def prepare_corpus(self, cards_csv, filename):
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
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        corpus = pd.DataFrame(columns=['card'])
        for _, row in df.iterrows():
            text = ''
            for feature, token in feature_dict.items():
                value = row.get(feature, '<empty>')
                text += f' {token} {value} {token[:1]}\\{token[1:]}'
            card = text.strip() + f' {special_tokens["eos"]}'
            corpus = corpus.append({'card': card}, ignore_index=True)
        
        corpus["tokens"] = corpus["card"].apply(lambda x: tokenizer.encode(x, ).ids)
        corpus["mc"] = corpus["card"].apply(lambda x: yoink(x, "mc"))
        corpus["power"] = corpus["card"].apply(lambda x: yoink(x, "power"))
        corpus["toughness"] = corpus["card"].apply(lambda x: yoink(x, "toughness"))
        corpus["cmc"] = corpus["mc"].apply(lambda x: sum([int(y) if y.isdigit() else 1 for y in x.replace(" ", "").replace("{", "").replace("}", "")]))

        corpus["power"] = corpus["power"].apply(lambda x: int(x) if x.isdigit() else x)
        corpus["toughness"] = corpus["toughness"].apply(lambda x: int(x) if x.isdigit() else x)
            

        corpus.to_csv(f'{filename}.csv', index=False)

        print(f"Corpus saved to {filename}")


def collate_fn(batch):
        def parse(text):
            parsed_list = ast.literal_eval(text)
            if parsed_list[0] != 1:
                parsed_list = [1] + parsed_list[:-1]
            return parsed_list
        ids = [parse(x[1]) for x in batch]
        ids = torch.tensor(ids)
        originals = [x[0] for x in batch]
        mc = [x[2] for x in batch]
        power = [x[3] for x in batch]
        toughness = [x[4] for x in batch]
        cmc = [x[5] for x in batch]

        return {
            "ids": ids, 
            "originals": originals,
            "mc": mc,
            "power": power,
            "toughness": toughness,
            "cmc": cmc,
            }

def get_dataloaders(test_set_portion, seed, batch_size):
    # DataLoader
    dataset = MagicCardDataset(target="data/labeled.csv")

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