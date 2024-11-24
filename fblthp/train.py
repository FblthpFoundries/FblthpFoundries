from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from TransformerVAE import TransformerVAE
import wandb
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm  # Import tqdm for progress bars
import json, requests, os
import traceback
import pandas
from torch.cuda.amp import autocast, GradScaler
from torch.cuda import memory_allocated, memory_reserved, memory_summary
from transformers import get_linear_schedule_with_warmup

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
class MagicCardDataset(Dataset):
    def __init__(self, tokenizer, max_len=125):

        self.tokenizer = tokenizer
        self.max_len = max_len
        if not os.path.exists('cards.csv'):
            self.pull_scryfall_to_csv('cards.csv')
        if not os.path.exists('corpus.csv'):
            self.prepare_corpus('cards.csv', 'corpus.csv')

        self.corpus_dataframe = pandas.read_csv('corpus.csv')

    def __len__(self):
        return len(self.corpus_dataframe)

    def __getitem__(self, idx):
        card_text = self.corpus_dataframe.iloc[idx]['card']

        tokenized = self.tokenizer.encode(card_text).ids
        sos_token_id = self.tokenizer.token_to_id('<sos>') if '<sos>' in self.tokenizer.get_vocab() else 1
        tokenized = [sos_token_id] + tokenized
        return torch.tensor(tokenized)
    
    def get_raw(self, idx):
        card_text = self.corpus_dataframe.iloc[idx]['card']
        return card_text

    def pull_scryfall_to_csv(self, filename):
        r = requests.get('https://api.scryfall.com/bulk-data/oracle-cards')

        assert(r.status_code == 200)

        r.close()

        jsonURI = json.loads(r.content)['download_uri']

        r = requests.get(jsonURI)

        cards = json.loads(r.content)
        r.close()

        #https://scryfall.com/docs/api/cards Currently only loading creatures

        features = ['mana_cost', 'name', 'type_line', 'power', 'toughness', 'oracle_text', 'loyalty', 'flavor_text']

        cardNums = len(cards)

        num = 0

        f = open(filename, 'w', encoding='utf-8')

        data = ""
        for feature in features:
            data += feature + ','
        data = data[:-1]

        f.write(data)

        for card in cards:
            num  += 1
            if  'Token' in card['type_line'] or 'card_faces' in card:
                continue
            if not 'paper' in card['games']:
                continue
            if 'Hero' in card['type_line'] or 'Plane' in card['type_line'] or 'Card' in card['type_line']:
                continue
            data = '\n'
            for feature in features:
                if feature not in card:
                    data += '<empty>,'
                    continue
                data += '\"' + card[feature].replace("\"", "").replace('\n', ' <nl> ').replace('}{', '} {') + '\",'
            data = data[:-1]
            f.write(data)

            if num % 1000 == 0:
                print(f'{num}/{cardNums}')

        f.close()

    def prepare_corpus(self, cards_csv, filename):
        df = pandas.read_csv(cards_csv)
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
            corpus.append(text[1:] + self.specialTokenDict['eos'])

        f = open(filename, 'w', encoding='utf-8')
        f.write('card\n')
        for text in corpus:
            f.write('\"'+text + '\"\n')
        f.close()
        return corpus

def log_memory():
    allocated = memory_allocated() / (1024 ** 2)  # Convert to MB
    reserved = memory_reserved() / (1024 ** 2)    # Convert to MB
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")



def compute_beta(step, beta_start, beta_end, warmup_steps):
    """
    Compute the KL weight (beta) based on the current step.
    Args:
        step (int): Current training step.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.
        warmup_steps (int): Number of steps for annealing.
    Returns:
        float: Annealed beta value.
    """
    if step >= warmup_steps:
        return beta_end
    return beta_start + (beta_end - beta_start) * (step / warmup_steps)

def train_vae(model, dataloader, optimizer, tokenizer, num_steps=50000, device='cuda', max_len=125, decay_rate=0.9995,
              batch_size=8, lr_warmup_steps=5000, beta_start=0.0, beta_end=0.07, beta_warmup_steps=5000, 
              initial_teacher_forcing_ratio=1.0, final_teacher_forcing_ratio=0.1, free_bits=0.2):
    model.train()
    scaler = GradScaler()

    # Initialize the learning rate scheduler outside the loop
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=num_steps
    )

    # Log hyperparameters
    wandb.config.update({
        "learning_rate": optimizer.defaults["lr"],
        "total_steps": num_steps,
        "batch_size": batch_size,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "beta_warmup_steps": beta_warmup_steps,
    })

    # Initialize a single tqdm progress bar for the entire training process
    with tqdm(total=num_steps, unit="step") as pbar:
        step = 0
        data_iter = iter(dataloader)  # Create a single iterator for the dataloader
        try:
            while step < num_steps:
                try:
                    x = next(data_iter)  # Get the next batch
                except StopIteration:
                    # Restart the dataloader when it runs out of batches
                    data_iter = iter(dataloader)
                    x = next(data_iter)

                x = x.to(device)
                # Compute the teacher forcing ratio
                teacher_forcing_ratio = max(
                    final_teacher_forcing_ratio, 
                    initial_teacher_forcing_ratio * (decay_rate ** step)
                )

                optimizer.zero_grad()

                # Forward pass through the VAE
                with autocast():
                    decoded_x, logits, mu, logvar = model(x, target_seq=x, teacher_forcing_ratio=teacher_forcing_ratio)
                    beta = compute_beta(step, beta_start, beta_end, beta_warmup_steps)
                    loss, ce_loss, kl_loss, scaled_kld_loss = model.vae_loss(logits, x, mu, logvar, kl_weight=beta, free_bits=free_bits)

                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Step the learning rate scheduler
                scheduler.step()

                # Log stats to wandb for this batch
                wandb.log({
                    "loss": loss.item(),
                    "step": step,
                    "teacher_forcing_ratio": teacher_forcing_ratio,
                    "lr": scheduler.get_last_lr()[0],  # Log the learning rate
                    "ce_loss": ce_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "scaled_kl_loss": scaled_kld_loss.item(),
                    "beta": beta,
                })

                # Update tqdm with the current step and loss
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

                # Log additional information at intervals
                if step % 100 == 0:
                    log_generated_card(model, tokenizer, device, n=10)
                    log_memory()

                step += 1  # Increment step counter

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            filename = f"canceled_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            save_state(model, optimizer, scheduler, None, step, filename)
            print(f"Checkpoint saved successfully to {filename}. Exiting.")

        print("Training complete!")
        filename = f"final_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        save_state(model, optimizer, scheduler, step, filename)
        print(f"Final checkpoint saved successfully to {filename}.")

        # Finish the run (optional)
        wandb.finish()

# Function to generate and log a card to wandb
def log_generated_card(model, tokenizer, device='cuda', n=1):
    model.eval()  # Set model to evaluation mode
    for i in range(n):
        # Generate a random latent vector (or sample from the latent space)
        latent_vector = torch.randn(1, model.encoder.fc_mu.out_features).to(device)
        
        # Generate a card from the latent vector
        generated_token_ids = model.generate(latent_vector, max_len=50)[0]  # Adjust max_len if needed
        # Convert token IDs back to human-readable text
        generated_text = tokenizer.decode(generated_token_ids.squeeze().tolist(), skip_special_tokens=False)
        
        # Log the generated card text to wandb
        #wandb.log({"generated_card": generated_text})
        print("ids[:20]:", generated_token_ids.squeeze().tolist()[:20])
        print("decoded text: ", generated_text.replace('<pad>', '').replace('<eos>', ''))

    model.train()  # Switch back to training mode

def save_state(vae_model, optimizer, scheduler, step, filename):
    checkpoint = {
        "step": step,
        "model_state": vae_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(checkpoint, os.path.join("checkpoints", filename))
def load_state(checkpoint_path, vae_model, optimizer, scheduler):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Restore the model state
    vae_model.load_state_dict(checkpoint["model_state"])
    
    # Restore the optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    # Restore the scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    
    # Restore the RNG states for reproducibility
    torch.set_rng_state(checkpoint["rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    
    # Return the epoch and step to resume training
    return checkpoint["epoch"], checkpoint["step"]

# Main training script
def main():
    # Hyperparameters
    embed_dim = 256  # Embedding dimension
    num_heads = 8  # Number of attention heads
    hidden_dim = 2048  # Feedforward network hidden dimension
    num_layers = 6  # Number of transformer encoder/decoder layers
    max_len = 125  # Maximum length of sequences
    batch_size = 16 # Batch Size
    accumulation_steps = 4 # Gradient accumulation steps
    dropout = 0.1 # Dropout rate

    num_steps = 50000   # Number of training steps

    learning_rate = 0.00025  # Learning rate (maximum)

    initial_teacher_forcing_ratio = 1.0 # Initial teacher forcing ratio
    final_teacher_forcing_ratio = 0.1 # Final teacher forcing ratio
    decay_rate = 0.9995 # Decay rate for teacher forcing ratio

    beta_start = 0.0
    beta_end = 0.07
    beta_warmup_steps = 5000

    free_bits = 0.2

    lr_warmup_steps=5000


    vocab_size = 20000
    tokenizer_exists = os.path.exists("wordpiece_tokenizer.json")

    if not tokenizer_exists:
        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        special_tokens.extend([
            "<tl>","<name>","<mc>","<ot>","<power>","<toughness>","<loyalty>","<ft>", "<nl>",
            "<\\tl>", "<\\name>", "<\\mc>", "<\\ot>", "<\\power>", "<\\toughness>", "<\\loyalty>", "<\\ft>",
            "{W}", "{U}", "{B}", "{R}", "{G}", "{C}", "{X}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", 
            "{8}", "{9}", "{10}", "{11}", "{12}", "{13}", "{14}", "{15}", "+1/+1", "{T}"
        ])

        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

        tokenizer.enable_padding(pad_id=0, pad_token=special_tokens[0], length=max_len)
        tokenizer.enable_truncation(max_length=max_len)
        
        dataset = MagicCardDataset(tokenizer, max_len=max_len) # just here in order to make corpus.csv on a first go (bad code but it works)
        tokenizer.train(['corpus.csv'], trainer)
        tokenizer.save("wordpiece_tokenizer.json")
    else:
        tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")

    wandb.init(project="TransformerVAE", config={
    "learning_rate": learning_rate,
    "steps": num_steps,
    "batch_size": batch_size,
    "embed_dim": embed_dim,
    "num_heads": num_heads,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "max_len": max_len,
    })
    # Create the VAE model
    vae_model = TransformerVAE(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len, dropout=dropout)
    vae_model.print_model_size()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_model = vae_model.to(device)
    
    
    
    # DataLoader (use a dummy dataset for now)
    dataset = MagicCardDataset(tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    
    # Optimizer
    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

    # Start training
    try:
        train_vae(vae_model, dataloader, optimizer, tokenizer, num_steps=num_steps, device=device, max_len=max_len, decay_rate=decay_rate, \
                  batch_size=batch_size, lr_warmup_steps=lr_warmup_steps, beta_start=beta_start, beta_end=beta_end, beta_warmup_steps=beta_warmup_steps, \
                    initial_teacher_forcing_ratio=initial_teacher_forcing_ratio, final_teacher_forcing_ratio=final_teacher_forcing_ratio, free_bits=free_bits)
    except Exception as e:
        print(e)
        traceback.print_exc()
if __name__ == "__main__":
    main()
