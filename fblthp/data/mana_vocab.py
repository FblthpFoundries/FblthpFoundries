from typing import List, Set
import torch
import json
import pandas as pd
import re

class ManaVocabulary:
    def __init__(self):
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        
        self.symbol_to_idx = {}
        self.idx_to_symbol = {}

        self.build_vocabulary()
        
    def build_vocabulary(self):
        """
        Build vocabulary for mana symbols.
        Expected symbols like: {'W', 'U', 'B', 'R', 'G', '1', '2', 'X', etc.}
        """

        mana_symbols = {
            "{W}", "{U}", "{B}", "{R}", "{G}", "{C}", 
            "{0}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", 
            "{7}", "{8}", "{9}", "{10}", "{11}", "{12}", "{13}", 
            "{14}", "{15}", "{16}", "{17}", "{18}", "{19}", "{20}",
            "{X}", "{Y}", "{Z}", "{S}", "{100}", "{1000000}", "{W/U}",
            "{W/B}", "{B/R}", "{B/G}", "{U/B}", "{U/R}", "{R/G}", "{R/W}",
            "{G/W}", "{G/U}", "{2/W}", "{2/U}", "{2/B}", "{2/R}", "{2/G}",
            "{W/P}", "{U/P}", "{B/P}", "{R/P}", "{G/P}", "{C/P}", "{HW}", "{HR}",
            "{C/W}", "{C/U}", "{C/B}", "{C/R}", "{C/G}", "{B/G/P}", "{B/R/P}",
            "{G/W/P}", "{G/U/P}", "{R/W/P}", "{R/G/P}", "{W/B/P}", "{W/U/P}",
            "{U/B/P}", "{U/R/P}", "{D}"
        }




        # Start with special tokens
        self.symbol_to_idx = self.special_tokens.copy()
        current_idx = len(self.special_tokens)
        
        # Add mana symbols
        for symbol in sorted(mana_symbols):
            self.symbol_to_idx[symbol] = current_idx
            current_idx += 1
            
        # Create reverse mapping
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}
        
    def encode(self, mana_cost: str, pad_to_length: int = None) -> torch.Tensor:
        """
        Convert mana cost string to tensor of indices
        Example: "{2}{W}{U}" -> tensor([1, 7, 4, 5, 2])  # with BOS and EOS
        
        Args:
            mana_cost: String representation of mana cost
            pad_to_length: Optional length to pad sequence to
        """
        # Parse mana cost string into symbols
        symbols = self._parse_mana_string(mana_cost)
        
        # Convert to indices
        indices = [self.special_tokens['<BOS>']]
        for symbol in symbols:
            indices.append(self.symbol_to_idx.get(symbol, self.special_tokens['<UNK>']))
        indices.append(self.special_tokens['<EOS>'])
        
        # Pad if necessary
        if pad_to_length is not None:
            padding_needed = pad_to_length - len(indices)
            if padding_needed > 0:
                indices.extend([self.special_tokens['<PAD>']] * padding_needed)
            elif padding_needed < 0:
                indices = indices[:pad_to_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """
        Convert tensor of indices back to mana cost string
        """
        symbols = []
        for idx in indices.tolist():
            symbol = self.idx_to_symbol.get(idx, '<UNK>')
            if symbol not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                symbols.append(symbol)
        
        return ''.join(symbols)
    
    def _parse_mana_string(self, mana_cost: str) -> List[str]:
        """
        Parse mana cost string into list of symbols
        Example: "{2}{W}{U}" -> ["2", "W", "U"]
        """
        # Remove braces and split
        cleaned = re.findall(r'\{[^}]*\}', mana_cost)
        return cleaned
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary"""
        return len(self.symbol_to_idx)
    
    def save(self, path: str):
        """Save vocabulary to JSON file"""
        vocab_data = {
            'symbol_to_idx': self.symbol_to_idx,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load(self, path: str):
        """Load vocabulary from JSON file"""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
            
        self.symbol_to_idx = vocab_data['symbol_to_idx']
        self.special_tokens = vocab_data['special_tokens']
        
        # Recreate reverse mapping
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}

# Example usage:
if __name__ == "__main__":
    # Initialize vocabulary
    mana_vocab = ManaVocabulary()

    df = pd.read_csv('./data/labeled.csv')
    mana_costs = df['mc'].tolist()
    mana_costs = [mana_cost for mana_cost in mana_costs if isinstance(mana_cost, str)]
    mana_costs = list(set(mana_costs))
    max_length = 0

    for cost in mana_costs:
        cost = cost.replace(" ", "")
        encoded = mana_vocab.encode(cost, pad_to_length=None)
        length = len(encoded)
        if length > max_length:
            max_length = length
        decoded = mana_vocab.decode(encoded)
        if cost != decoded:
            print(f"Original: {cost}")
            print(f"Encoded: {encoded}")
            print(f"Decoded: {decoded}")
            print()
    print(max_length)
    
    # Example encoding/decoding
    mana_cost = "{2}{W}{U}{2}{3}{4}{4}{5}"
    encoded = mana_vocab.encode(mana_cost, pad_to_length=17)
    decoded = mana_vocab.decode(encoded)
    
    print(f"Vocabulary size: {mana_vocab.vocab_size}")
    print(f"Original mana cost: {mana_cost}")
    print(f"Encoded (with padding): {encoded}")
    print(f"Decoded: {decoded}")