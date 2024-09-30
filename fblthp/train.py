import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from TransformerVAE import TransformerVAE
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm  # Import tqdm for progress bars
import os
import pandas

# Dummy dataset for demonstration purposes
class RandomDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random sequence of token IDs
        torch.manual_seed(idx)
        return torch.randint(0, self.vocab_size, (self.seq_length,))
class MagicCardDatasetProcessor():
    def __init__(self):
        self.featureDict ={
        'type_line': '<tl>',
        'name': '<name>',
        'mana_cost': '<mc>',
        'oracle_text': '<ot>',
        'power': '<power>',
        'toughness': '<toughness>',
        'loyalty' : '<loyalty>',
        'flavor_text': '<ft>',
        
        }

        self.specialTokenDict = {
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

        self.pp = '\\+[0-9|X]+/\\+[0-9|X]+'
        self.mm = '\\-[0-9|X]+/\\-[0-9|X]+'
        self.xx = '[0-9|X]+/[0-9|X]+'
        self.pm = '\\+[0-9|X]+/\\-[0-9|X]+'
        self.mp = '\\-[0-9|X]+/\\+[0-9|X]+'

    def sanitize(self, text):
        while '{' in text:
            indexOpen  = text.index('{')
            indexClose = text.index('}')

            symbol =text[indexOpen: indexClose + 1]
            token = '<' + text[indexOpen + 1: indexClose] + '>'

            if not symbol in self.specialTokenDict:
                self.specialTokenDict[symbol] = token

            text = text[: indexOpen] + token + text[indexClose + 1:]

        return text

    def prepare_corpus(self, csv):
        df = pandas.read_csv(csv)

        

        corpus = []

        for index, row in df.iterrows():
            text= ''
            name = row['name']
            for feature in self.featureDict:
                append = ' ' + self.featureDict[feature] 
                if feature in row:
                    append += ' ' + str(row[feature]) if not str(row[feature]) == '<empty>' else '' 
                if not feature == 'name':
                    append = append.replace(name, '~')
                text +=  append + ' ' + self.featureDict[feature][:1] + '\\' + self.featureDict[feature][1:]
            corpus.append(self.sanitize(text[1:]) + self.specialTokenDict['eos'])

        f = open('corpus.csv', 'w', encoding='utf-8')
        f.write('card\n')
        for text in corpus:
            f.write('\"'+text + '\"\n')
        f.close()
        return corpus


# Training loop with tqdm progress bars
def train_vae(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Wrap the dataloader with tqdm for batch-level progress bar
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            for x in tepoch:
                x = x.to(device)
                optimizer.zero_grad()
                #print(f"X Shape {x.shape}")
                # Forward pass through the VAE
                decoded_x, logits, mu, logvar = model(x)
                #print(f"Decoded X Shape {decoded_x.shape}")
                # Compute the VAE loss
                loss = model.vae_loss(logits, x, mu, logvar)
                
                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
                
                # Accumulate the loss
                total_loss += loss.item()
                
                # Update tqdm with current loss
                tepoch.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')


# Main training script
def main():
    # Hyperparameters
    embed_dim = 128  # Embedding dimension
    num_heads = 1  # Number of attention heads
    hidden_dim = 256  # Feedforward network hidden dimension
    num_layers = 1  # Number of transformer encoder/decoder layers
    max_len = 100  # Maximum length of sequences
    batch_size = 4
    num_epochs = 4
    learning_rate = 0.001

    processor = MagicCardDatasetProcessor()
    corpus = processor.prepare_corpus('cards.csv')

    vocab_size = 20000
    tokenizer_exists = False

    if not tokenizer_exists:
        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.train(['corpus.csv'], trainer)
        tokenizer.save("wordpiece_tokenizer.json")
    else:
        tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")


    # Create the VAE model
    vae_model = TransformerVAE(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len)
    vae_model.print_model_size()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_model = vae_model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
    
    # DataLoader (use a dummy dataset for now)
    dataset = RandomDataset(num_samples=1000, seq_length=max_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Start training
    train_vae(vae_model, dataloader, optimizer, num_epochs=num_epochs, device=device)
    torch.save(vae_model.state_dict(), 'vae_model.pt')

if __name__ == "__main__":
    main()
