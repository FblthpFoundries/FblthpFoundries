"""
Magic: The Gathering Card Dataset Processor

This module provides functionality to download, process, and prepare Magic card data
from Scryfall API for use in machine learning models.
"""

import os
import ast
import re
import logging
from typing import Tuple, Dict, List, Optional, Union, Any

from .mana_vocab import ManaVocabulary
from .tokenizers_help import get_mtg_tokenizer

import pandas as pd
import requests
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DATA_DIR, 'datasets')
os.makedirs(DATASET_DIR, exist_ok=True)
DEFAULT_TOKENIZER_PATH = os.path.join(DATASET_DIR, "wordpiece_tokenizer.json")
DEFAULT_CARDS_PATH = os.path.join(DATASET_DIR, 'cards.csv')
DEFAULT_LINES_PATH = os.path.join(DATASET_DIR, 'lines.txt')
DEFAULT_LABELED_PATH = os.path.join(DATASET_DIR, 'labeled.csv')
MOMIR_LINES_PATH = os.path.join(DATASET_DIR, "momir_lines.txt")

# Token sequence lengths determined from analysis
TOKEN_LENGTHS = {
    'mana': 18,
    'oracle': 90, 
    'name': 20, 
    'type': 15, 
    'flavor_text': 50
}

# Feature mappings for card data
FEATURE_DICT = {
    'type_line': '<tl>',
    'name': '<name>',
    'mana_cost': '<mc>',
    'oracle_text': '<ot>',
    'power': '<power>',
    'toughness': '<toughness>',
    'loyalty': '<loyalty>',
    'flavor_text': '<ft>',
}

SPECIAL_TOKENS = {
    'eos': '<eos>',
    'pad': '<pad>'
}



class MagicCardDataset(Dataset):
    """Dataset class for Magic: The Gathering cards.
    
    This dataset handles downloading card data from Scryfall API,
    preparing and processing the data, and providing access to
    tokenized card text for training machine learning models.
    """

    def __init__(self, 
                 max_len: int = 125, 
                 target: str = DEFAULT_LABELED_PATH, 
                 prepare_corpus: bool = False,
                 tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
                 cards_path: str = DEFAULT_CARDS_PATH):
        """Initialize the Magic Card Dataset.
        
        Args:
            max_len: Maximum length of token sequences
            target: Path to the labeled CSV file
            prepare_corpus: Whether to download and prepare data if it doesn't exist
            tokenizer_path: Path to the tokenizer JSON file
            cards_path: Path to the cards CSV file
        """
        self.tokenizer_path = "wordpiece_tokenizer.json"
        self.max_len = max_len
        self.cards_path = cards_path
        
        if prepare_corpus:
            self.prepare_corpus()


        try:
            self.corpus_dataframe = pd.read_csv(target)
        except Exception as e:
            logger.error(f"Failed to load dataset from {target}: {e}")
            raise

    def __len__(self) -> int:
        """Return the number of items in the dataset.
        
        Returns:
            Number of items in the dataset
        """
        return len(self.corpus_dataframe)

    def __getitem__(self, idx: int) -> List[Any]:
        """Get a specific item from the dataset by index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            List of values for the specified card
        """
        row = self.corpus_dataframe.iloc[idx]
        return row.tolist()
    
    def prepare_corpus(self) -> None:
        """Prepare the corpus by downloading and processing card data if needed."""
        if not os.path.exists(DEFAULT_CARDS_PATH):
            self.pull_scryfall_to_csv()
                
        if not os.path.exists(DEFAULT_LABELED_PATH):
            self._prepare_corpus()

    @staticmethod
    def pull_scryfall_to_csv() -> None:
        """Pull card data from Scryfall API and save to CSV.
        
        Args:
            filename: Path to save the card data CSV
        """
        logger.info("Pulling data from Scryfall API...")
        try:
            # Get the bulk data URI
            response = requests.get('https://api.scryfall.com/bulk-data/oracle-cards')
            response.raise_for_status()
            
            json_uri = response.json()['download_uri']
            
            # Download the card data with progress bar
            logger.info(f"Downloading cards from {json_uri}")
            response = requests.get(json_uri, stream=True)
            response.raise_for_status()
            
            # Save the JSON response to memory
            card_data = response.json()
            
            features = ['mana_cost', 'name', 'type_line', 'power', 'toughness', 
                       'oracle_text', 'loyalty', 'flavor_text']
            
            
            # Process and save the cards to CSV with progress bar
            logger.info(f"Processing {len(card_data)} cards...")
            
            with open(DEFAULT_CARDS_PATH, 'w', encoding='utf-8') as f:
                f.write(','.join(features) + '\n')
                
                for card in tqdm(card_data, desc="Processing cards"):
                    # Skip tokens, double-faced cards, and non-paper cards
                    if ('Token' in card.get('type_line', '') or 
                        'card_faces' in card or 
                        'paper' not in card.get('games', [])):
                        continue
                    
                    row = []
                    for feature in features:
                        # Clean up the text
                        thing = card.get(feature, "<empty>")
                        thing = str(thing).replace("\"", "").replace("\n", " <nl> ").replace("}{", "} {")
                        row.append(f'"{thing}"')
                        
                    f.write(','.join(row) + '\n')
            logger.info(f"Card data saved to {DEFAULT_CARDS_PATH}")

            with open(DEFAULT_LINES_PATH, 'w', encoding='utf-8') as f:
                
                for card in tqdm(card_data, desc="Processing cards"):
                    # Skip tokens, double-faced cards, and non-paper cards
                    if ('Token' in card.get('type_line', '') or 
                        'card_faces' in card or 
                        'paper' not in card.get('games', [])):
                        continue
                    
                    row = []
                    for feature in features:
                        # Clean up the text
                        thing = card.get(feature, "<empty>")
                        thing = str(thing).replace("\"", "").replace("\n", " <nl> ").replace("}{", "} {")
                        row.append(f'{FEATURE_DICT[feature]}{thing}<\\{FEATURE_DICT[feature][1:]}')
                        
                    f.write(''.join(row) + '\n')
            logger.info(f"Card data saved to {DEFAULT_LINES_PATH}")




            mv = ManaVocabulary()
            #Momir Lines
            with open(MOMIR_LINES_PATH, 'w', encoding='utf-8') as f:
                
                for card in tqdm(card_data, desc="Processing cards"):
                    # Skip tokens, double-faced cards, and non-paper cards
                    if ('Token' in card.get('type_line', '') or 
                        'card_faces' in card or 
                        'paper' not in card.get('games', [])):
                        continue
                    
                    row = []
                    row.append(f'<mv>{mv.mana_value(card.get("mana_cost", ""))}<\\mv>')
                    for feature in features:
                        # Clean up the text
                        thing = card.get(feature, "<empty>")
                        thing = str(thing).replace("\"", "").replace("\n", " <nl> ").replace("}{", "} {")
                        row.append(f'{FEATURE_DICT[feature]}{thing}<\\{FEATURE_DICT[feature][1:]}')
                    if 'Creature' not in card.get("type_line", ""):
                        continue
                    f.write(''.join(row) + '\n')
            logger.info(f"Card data saved to {MOMIR_LINES_PATH}")
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing JSON data: {e}")
            raise
        except IOError as e:
            logger.error(f"Error writing to file: {e}")
            raise

    def _extract_feature(self, text: str, attribute: str) -> Optional[str]:
        """Extract a feature from text using regex.
        
        Args:
            text: The text to search
            attribute: The attribute to extract
            
        Returns:
            The extracted text or None if not found
        """
        pattern = fr"<{attribute}>(.*?)<\\{attribute}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None

    def _prepare_corpus(self) -> None:
        """Process raw card data into a tokenized corpus.
        
        Args:
            cards_csv: Path to the raw cards CSV
            output_filename: Path to save the processed corpus
        """
        logger.info("Preparing corpus...")
        
        df = pd.read_csv(DEFAULT_CARDS_PATH)
        
        # Initialize tokenizers
        text_tokenizer = get_mtg_tokenizer()
        
        # Create tokenizers for each field with appropriate padding
        tokenizers = {}
        for field, length in [
            ('oracle', TOKEN_LENGTHS['oracle']), 
            ('name', TOKEN_LENGTHS['name']),
            ('type', TOKEN_LENGTHS['type']), 
            ('flavor', TOKEN_LENGTHS['flavor_text'])
        ]:
            tokenizers[field] = text_tokenizer
            tokenizers[field].enable_padding(
                pad_id=0, 
                pad_token=SPECIAL_TOKENS['pad'], 
                length=length
            )
            tokenizers[field].enable_truncation(max_length=length)

        mana_tokenizer = ManaVocabulary()
        
        # Process data with progress bar
        corpus = pd.DataFrame(columns=['card'])
        
        logger.info(f"Processing {len(df)} cards into corpus...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing corpus"):
            text = ''
            for feature, token in FEATURE_DICT.items():
                value = row.get(feature, '')
                text += f' {token} {value} {token[:1]}\\{token[1:]}'
            card = text.strip() + f' {SPECIAL_TOKENS["eos"]}'
            corpus.loc[len(corpus)] = [card]
        
        # Extract and tokenize fields
        logger.info("Tokenizing card fields...")
        corpus["tokens"] = corpus["card"].apply(lambda x: text_tokenizer.encode(x).ids)
        
        # Extract fields
        for field, tag in [
            ('mc', 'mc'), ('power', 'power'), ('toughness', 'toughness'),
            ('oracle_text', 'ot'), ('name', 'name'), 
            ('type_line', 'tl'), ('flavor_text', 'ft')
        ]:
            corpus[field] = corpus["card"].apply(lambda x: self._extract_feature(x, tag))
        
        # Process numeric fields
        corpus["cmc"] = corpus["mc"].apply(
            lambda x: sum([int(y) if y.isdigit() else 1 
                            for y in str(x).replace(" ", "").replace("{", "").replace("}", "")]) 
            if x else 0
        )
        
        # Convert power/toughness to int when possible
        for field in ['power', 'toughness']:
            corpus[field] = corpus[field].apply(
                lambda x: int(x) if x and str(x).isdigit() else x
            )

        # Tokenize text fields
        for field in ['oracle_text', 'name', 'type_line', 'flavor_text']:
            token_field = f"{field.split('_')[0]}_tokens"
            corpus[token_field] = corpus[field].apply(
                lambda x: tokenizers[field.split('_')[0]].encode(str(x) if x else "").ids
            )
        
        # Tokenize mana cost
        corpus["mana_tokens"] = corpus["mc"].apply(
            lambda x: mana_tokenizer.encode(str(x) if x else "", 
                                            pad_to_length=TOKEN_LENGTHS['mana']).tolist()
        )

        # Save the processed corpus
        corpus.to_csv(DEFAULT_LABELED_PATH, index=False)
        logger.info(f"Corpus saved to {DEFAULT_LABELED_PATH}")








def parse_token_list(text: str) -> List[int]:
    """Parse a string representation of a token list.
    
    Ensures the first token is 1 (likely the BOS token).
    
    Args:
        text: String representation of a token list
        
    Returns:
        List of token IDs
    """
    try:
        parsed_list = ast.literal_eval(text)
        # Ensure the first token is 1 (BOS token)
        if parsed_list and parsed_list[0] != 1:
            parsed_list = [1] + parsed_list[:-1]
        return parsed_list
    except (SyntaxError, ValueError) as e:
        logger.error(f"Failed to parse token list: {e}")
        return [1]  # Return default if parsing fails


def collate_fn(batch: List[List]) -> Dict[str, Any]:
    """Collate function for DataLoader to process a batch of samples.
    
    Args:
        batch: Batch of samples from the dataset
        
    Returns:
        Dictionary of tensors and values
    """
    try:
        # Extract fields
        ids = [parse_token_list(x[1]) for x in batch]
        ids = torch.tensor(ids)
        
        # Extract all other fields
        field_names = [
            "originals", "mc", "power", "toughness", "cmc", 
            "oracle_text", "name", "type_line", "flavor_text"
        ]
        fields = {field_names[i]: [x[i] for x in batch] for i in range(len(field_names))}
        
        # Extract and parse token fields
        token_fields = [
            "oracle_tokens", "name_tokens", "type_line_tokens", 
            "flavor_text_tokens", "mana_tokens"
        ]
        parsed_tokens = {
            field: [parse_token_list(x[i+10]) for x in batch] 
            for i, field in enumerate(token_fields)
        }
        
        # Combine all fields
        result = {"ids": ids, **fields, **parsed_tokens}
        return result
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        raise


def get_dataloaders(
    test_set_portion: float = 0.2, 
    seed: int = 42, 
    batch_size: int = 32,
    target_path: str = DEFAULT_LABELED_PATH,
    prepare_corpus: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create training and test DataLoaders for the Magic card dataset.
    
    Args:
        test_set_portion: Portion of data to use for testing (0.0-1.0)
        seed: Random seed for reproducibility
        batch_size: Batch size for DataLoaders
        target_path: Path to labeled CSV file
        prepare_corpus: Whether to download and prepare data if it doesn't exist
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    logger.info("Creating dataloaders...")

    # Create dataset
    dataset = MagicCardDataset(target=target_path, prepare_corpus=prepare_corpus)

    # Split the dataset into training and validation sets
    train_size = int(len(dataset) * (1 - test_set_portion))
    test_size = len(dataset) - train_size
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=2, 
        persistent_workers=True, 
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=2, 
        persistent_workers=True, 
        collate_fn=collate_fn
    )

    logger.info(f"Created dataloaders - Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    return train_dataloader, test_dataloader


def analyze_sequence_lengths(dataloader: DataLoader) -> None:
    """Analyze and visualize sequence lengths in the dataset.
    
    Args:
        dataloader: DataLoader to analyze
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        logger.info("Analyzing sequence lengths...")
        
        # Fields to analyze
        fields = ["mana_tokens", "oracle_tokens", "name_tokens", "type_line_tokens", "flavor_text_tokens"]
        datas = [[] for _ in range(len(fields))]
        maxes = [0 for _ in range(len(fields))]
        max_items = [None for _ in range(len(fields))]
        
        # Process batches with progress bar
        for batch in tqdm(dataloader, desc="Analyzing sequences"):
            for i, field in enumerate(fields):
                # Calculate non-zero sequence length
                seq_len = torch.nonzero(torch.tensor(batch[field])).size(0)
                datas[i].append(seq_len)
                
                # Track maximum length and example
                if seq_len > maxes[i]:
                    maxes[i] = seq_len
                    max_items[i] = torch.tensor(batch[field])
        
        # Print maximum lengths
        logger.info(f"Maximum sequence lengths: {maxes}")
        
        # Plot histograms
        for i, (field, data) in enumerate(zip(fields, datas)):
            data_array = np.array(data)
            plt.figure(figsize=(10, 6))
            plt.hist(data_array, bins='auto', edgecolor='black')
            plt.title(f'Histogram of {field} Sequence Lengths')
            plt.xlabel('Sequence Length')
            plt.ylabel('Frequency')
            plt.savefig(f"{field}_histogram.png")
            plt.close()
            
        logger.info("Analysis complete. Histograms saved as PNG files.")
        
    except ImportError:
        logger.warning("Matplotlib or numpy not available. Skipping visualization.")
    except Exception as e:
        logger.error(f"Error during sequence length analysis: {e}")


if __name__ == "__main__":
    # Example usage
    train_dataloader, test_dataloader = get_dataloaders(
        test_set_portion=0.1,
        seed=42,
        batch_size=1,
        prepare_corpus=True
    )
    
    # Analyze sequence lengths
    #analyze_sequence_lengths(train_dataloader)
