import warnings
# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Using `TRANSFORMERS_CACHE` is deprecated.*"
)

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from TransformerVAE import TransformerVAE, BertVAE
import wandb
from transformers import BertTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm  # Import tqdm for progress bars
import json, requests, os
import traceback
import pandas
from torch.cuda.amp import autocast, GradScaler
from torch.cuda import memory_allocated, memory_reserved, memory_summary
from transformers import get_linear_schedule_with_warmup
import argparse


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
    def __init__(self, tokenizer, max_len=125, target="corpus.csv", get_raw=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_raw = get_raw
        if target == "corpus.csv":
            if not os.path.exists('cards.csv'):
                self.pull_scryfall_to_csv('cards.csv')
            if not os.path.exists('corpus.csv'):
                self.prepare_corpus('cards.csv', 'corpus.csv')

            self.corpus_dataframe = pandas.read_csv('corpus.csv')
        else:
            self.corpus_dataframe = pandas.read_csv(target)

    def __len__(self):
        return len(self.corpus_dataframe)

    def __getitem__(self, idx):
        rows = self.corpus_dataframe.iloc[idx]
        if self.get_raw:
            return rows
        card_text = rows['card']
        if not self.tokenizer:
            return card_text

        if isinstance(self.tokenizer, BertTokenizer):
            # For BertTokenizer
            tokenized = self.tokenizer.encode(card_text, add_special_tokens=True, max_length=self.max_len, truncation=True, padding='max_length')
        else:
            # For Tokenizer from the tokenizers library
            tokenized = self.tokenizer.encode(card_text).ids
            sos_token_id = self.tokenizer.token_to_id('<sos>') if '<sos>' in self.tokenizer.get_vocab() else 1
            tokenized = [sos_token_id] + tokenized[:self.max_len - 1]  # Truncate to max_len
            tokenized += [0] * (self.max_len - len(tokenized))  # Pad with 0s to max_len

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

# Function to generate and log a card to wandb
def log_generated_card(model, tokenizer, device='cuda', n=1):
    model.eval()  # Set model to evaluation mode
    for i in range(n):
        # Generate a random latent vector (or sample from the latent space)
        latent_vector = torch.randn(1, model.encoder.fc_mu.out_features).to(device)
        
        # Generate a card from the latent vector
        generated_token_ids = model.generate(latent_vector, max_len=125)[0]  # Adjust max_len if needed
        # Convert token IDs back to human-readable text
        generated_text = tokenizer.decode(generated_token_ids.squeeze().tolist(), skip_special_tokens=False)
        
        # Log the generated card text to wandb
        #wandb.log({"generated_card": generated_text})
        print("ids[:10]:", generated_token_ids.squeeze().tolist()[:10])
        print("decoded text: ", generated_text.replace('<pad>', '').replace('<eos>', ''))

    model.train()  # Switch back to training mode

def reconstruct_cards(model, tokenizer, dataloader, device='cuda', n=1):
    model.eval()  # Set model to evaluation mode
    reconstructed_texts = []
    original_texts = []
    
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            if i >= n:
                break
            x = x.to(device)
            x = x[0, :].unsqueeze(0) # take only the first one
            mu, logvar = model.encoder(x)
            z = model.reparameterize(mu, logvar)
            reconstructed_token_ids = model.generate(z, max_len=x.size(1))
            reconstructed_text = tokenizer.decode(reconstructed_token_ids.squeeze().tolist(), skip_special_tokens=False)
            original_text = tokenizer.decode(x.squeeze().tolist(), skip_special_tokens=False)
            reconstructed_texts.append(reconstructed_text)
            original_texts.append(original_text)
    
    for original, reconstructed in zip(original_texts, reconstructed_texts):
        print(f"Original: {original.replace('<pad>', '').replace('<eos>', '').replace('<sos>', '').strip()}")  # Remove padding tokens
        print()
        print(f"Reconstructed: {reconstructed.replace('<eos>', '').replace('<sos>', '').strip()}")  # Remove padding tokens
        print("-" * 50)
    
    model.train()  # Switch back to training mode

def save_state(vae_model, optimizer, scheduler, step, filename, hypers):
    checkpoint = {
        "step": step,
        "model_state": vae_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "hypers": hypers
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

def train_vae(model, dataloader, optimizer, scheduler, tokenizer, num_steps=50000, device='cuda', max_len=125, decay_rate=0.9995,
              batch_size=8, beta_start=0.0, beta_end=0.07, beta_warmup_steps=5000, 
               free_bits=0.2, test_dataloader=None, start_step=1, hypers={}):
    model.train()
    scaler = GradScaler()

    

    # Log hyperparameters
    wandb.config.update({
        "learning_rate": optimizer.defaults["lr"],
        "total_steps": num_steps,
        "batch_size": batch_size,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "beta_warmup_steps": beta_warmup_steps,
    })
    best_loss = 100000
    # Initialize a single tqdm progress bar for the entire training process
    with tqdm(total=num_steps, initial=start_step, unit="step") as pbar:
        step = start_step
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

                optimizer.zero_grad()

                # Forward pass through the VAE
                with autocast():
                    logits, mu, logvar = model(x, target_seq=x)
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
                if step % 2000 == 0:
                    print("-Triggering DIAGNOSTIC LOGGING! -_-")
                    if test_dataloader is not None:
                        model.eval()
                        with torch.no_grad():
                            l_total = 0.0
                            l_recon = 0.0
                            l_scaled = 0.0
                            l_kl = 0.0
                            for test_x in test_dataloader:
                                test_x = test_x.to(device)
                                test_logits, test_mu, test_logvar = model(test_x, target_seq=test_x)
                                total, recon, scaled, kl = model.vae_loss(test_logits, test_x, test_mu, test_logvar, kl_weight=beta, free_bits=free_bits)
                                l_total += total.item()
                                l_recon += recon.item()
                                l_scaled += scaled.item()
                                l_kl += kl.item()
                            l_total /= len(test_dataloader)
                            l_recon /= len(test_dataloader)
                            l_scaled /= len(test_dataloader)
                            l_kl /= len(test_dataloader)
                            wandb.log({
                                "val_loss": l_total,
                                "val_ce_loss": l_recon,
                                "val_kl_loss": l_kl,
                                "val_scaled_kl_loss": l_scaled,
                            })
                        model.train()
                        print("----------------[Validation Loss:]-----------------")
                        print(f"Total loss: {l_total:.4f}, CE loss: {l_recon:.4f}, KL loss: {l_kl:.4f}, Scaled KL loss: {l_scaled:.4f}")

                        if l_recon < best_loss:
                            best_loss = l_recon
                            save_state(model, optimizer, scheduler, step, f"best_checkpoint.pt", hypers=hypers)
                    print("----------[Sample cards:]----------------- ")
                    log_generated_card(model, tokenizer, device, n=10)
                    print("----------[Reconstructed cards:]----------------- ")
                    reconstruct_cards(model, tokenizer, test_dataloader, device, n=10)
                    print("----------[Memory Usage:]----------------- ")
                    log_memory()
                
                
                if step % 10000 == 0:
                    print("Saving checkpoint...")
                    save_state(model, optimizer, scheduler, step, f"checkpoint_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt", hypers=hypers)

                step += 1  # Increment step counter

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            filename = f"canceled_checkpoint_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt"
            save_state(model, optimizer, scheduler, step, filename, hypers=hypers)
            print(f"Checkpoint saved successfully to {filename}. Exiting.")
            return

        print("Training complete!")
        filename = f"final_checkpoint_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt"
        save_state(model, optimizer, scheduler, step, filename, hypers=hypers)
        print(f"Final checkpoint saved successfully to {filename}.")

        # Finish the run (optional)
        wandb.finish()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a VAE model (TransformerVAE or BertVAE).")
    parser.add_argument("--model", type=str, choices=["TransformerVAE", "BertVAE"], default="TransformerVAE",
                        help="Choose which VAE model to train.")
    parser.add_argument("--resume_weights", type=str, default=None,)
    args = parser.parse_args()

    # Common Hyperparameters
    embed_dim = 1024
    num_heads = 32
    hidden_dim = 2048
    num_encoder_layers = 4
    num_decoder_layers = 8
    max_len = 125
    batch_size = 64
    dropout = 0.1
    num_steps = 500000
    learning_rate = 0.00003
    decay_rate = 0.9995
    beta_start = 0.0
    beta_end = 0.01
    beta_warmup_steps = 10000
    free_bits = 0.2
    lr_warmup_steps = 10000
    vocab_size = 20000
    test_set_portion = 0.05
    seed = 42

    hypers = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "max_len": max_len,
        "batch_size": batch_size,
        "dropout": dropout,
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "decay_rate": decay_rate,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "beta_warmup_steps": beta_warmup_steps,
        "free_bits": free_bits,
        "lr_warmup_steps": lr_warmup_steps,
        "vocab_size": vocab_size,
        "test_set_portion": test_set_portion,
        "seed": seed
    }

    # Initialize or load tokenizer
    tokenizer_exists = os.path.exists("wordpiece_tokenizer.json")
    if not tokenizer_exists:
        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        special_tokens.extend([
            "<tl>", "<name>", "<mc>", "<ot>", "<power>", "<toughness>", "<loyalty>", "<ft>", "<nl>",
            "<\\tl>", "<\\name>", "<\\mc>", "<\\ot>", "<\\power>", "<\\toughness>", "<\\loyalty>", "<\\ft>",
            "{W}", "{U}", "{B}", "{R}", "{G}", "{C}", "{X}", "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", 
            "{8}", "{9}", "{10}", "{11}", "{12}", "{13}", "{14}", "{15}", "+1/+1", "{T}"
        ])
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.enable_padding(pad_id=0, pad_token=special_tokens[0], length=max_len)
        tokenizer.enable_truncation(max_length=max_len)
        dataset = MagicCardDataset(tokenizer, max_len=max_len)  # Generates corpus.csv
        tokenizer.train(["corpus.csv"], trainer)
        tokenizer.save("wordpiece_tokenizer.json")
    else:
        tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")

    # Initialize wandb
    wandb.init(project="VAE_Training", config={
        "learning_rate": learning_rate,
        "steps": num_steps,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "max_len": max_len,
    })

    # Choose model
    device = "cuda" if torch.cuda.is_available() else "cpu"


    if args.model == "TransformerVAE":
        model = TransformerVAE(vocab_size, embed_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len, dropout=dropout).to(device)
        train_function = train_vae  # Use the TransformerVAE training function
        step = 1
    
    model.print_model_size()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the learning rate scheduler outside the loop
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=num_steps
    )

    if args.resume_weights:

        print(f"Resuming from {args.resume_weights}")
        checkpoint = torch.load(args.resume_weights)

        vocab_size = checkpoint["hypers"]["vocab_size"]
        embed_dim = checkpoint["hypers"]["embed_dim"]
        num_heads = checkpoint["hypers"]["num_heads"]
        hidden_dim = checkpoint["hypers"]["hidden_dim"]
        if "num_layers" in checkpoint["hypers"]:
            num_decoder_layers = checkpoint["hypers"]["num_layers"]
            num_encoder_layers = checkpoint["hypers"]["num_layers"]
        else:
            num_decoder_layers = checkpoint["hypers"]["num_decoder_layers"]
            num_encoder_layers = checkpoint["hypers"]["num_encoder_layers"]
        max_len = checkpoint["hypers"]["max_len"]
        dropout = checkpoint["hypers"]["dropout"]
        num_steps = checkpoint["hypers"]["num_steps"]
        learning_rate = checkpoint["hypers"]["learning_rate"]
        beta_start = checkpoint["hypers"]["beta_start"]
        beta_end = checkpoint["hypers"]["beta_end"]
        beta_warmup_steps = checkpoint["hypers"]["beta_warmup_steps"]
        free_bits = checkpoint["hypers"]["free_bits"]
        test_set_portion = checkpoint["hypers"]["test_set_portion"]
        seed = checkpoint["hypers"]["seed"]
        step = checkpoint["step"]


        model = TransformerVAE(vocab_size, embed_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len, dropout=dropout).to(device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        


    # DataLoader
    dataset = MagicCardDataset(tokenizer, max_len=max_len)

    # Split the dataset into training and validation sets
    train_size = int(len(dataset) * (1 - test_set_portion))
    test_size = len(dataset) - train_size
    torch.manual_seed(seed) # Seed so the random split is the same and doesn't contaminate when reloading a model
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  pin_memory=True, num_workers=2, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                 pin_memory=True, num_workers=2, persistent_workers=True)

    

    # Train the model
    try:
        train_function(
            model, train_dataloader, optimizer, scheduler, tokenizer, num_steps=num_steps, device=device, max_len=max_len,
            decay_rate=decay_rate, batch_size=batch_size, beta_start=beta_start, beta_end=beta_end, 
            beta_warmup_steps=beta_warmup_steps,free_bits=free_bits, test_dataloader=test_dataloader, start_step=step, hypers=hypers
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()
if __name__ == "__main__":
    main()
