import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

# Import our custom modules
from data.datasets import get_dataloaders
from models.LightweightVAE import LightweightVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Train LightweightVAE for Magic: The Gathering card generation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--text_embed_dim", type=int, default=128)
    parser.add_argument("--mana_embed_dim", type=int, default=16)
    parser.add_argument("--kl_weight", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default="mtg-vae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--prepare_corpus", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def train_epoch(model, dataloader, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        reconstructions = model(batch)
        loss, loss_components = model.compute_loss(batch, reconstructions)

        #with torch.autograd.detect_anomaly():
        # Backward pass
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Update metrics
        batch_size = batch['name_tokens'].size(0)
        total_loss += loss.item() * batch_size
        
        # Log to wandb (simplified)
        wandb.log({
            'batch_loss': loss.item(),
            'batch_recon_loss': loss_components['recon_loss'].item(),
            'batch_kl_loss': loss_components['kl_loss'].item(),
        })
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            reconstructions = model(batch)
            loss, _ = model.compute_loss(batch, reconstructions)
            
            # Update metrics
            batch_size = batch['name_tokens'].size(0)
            total_loss += loss.item() * batch_size
    
    return total_loss / len(dataloader.dataset)

def main():
    # Parse arguments and set up environment
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Load datasets
    print("Loading datasets...")
    train_dataloader, val_dataloader = get_dataloaders(
        test_set_portion=0.1,
        batch_size=args.batch_size,
        seed=args.seed,
        prepare_corpus=args.prepare_corpus
    )
    
    # Create model with simplified vocabulary sizes
    sample_batch = next(iter(train_dataloader))
    print(sample_batch)
    model = LightweightVAE(
        main_vocab_size=30000,
        name_type_vocab_size=30000,
        mana_vocab_size=80,
        text_embed_dim=args.text_embed_dim,
        mana_embed_dim=args.mana_embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_oracle_len=sample_batch['oracle_tokens'].size(1),
        max_flavor_len=sample_batch['flavor_tokens'].size(1) if 'flavor_tokens' in sample_batch else 50,
        max_name_len=sample_batch['name_tokens'].size(1),
        max_type_len=sample_batch['type_tokens'].size(1),
        max_mana_len=sample_batch['mana_tokens'].size(1),
        kl_weight=args.kl_weight,
        debug_mode=False
    ).to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Track best model
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device, args.grad_clip)
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'epoch': epoch
        })
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': vars(args)
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"Saved best model with validation loss {best_val_loss:.4f}")
        
        # Save checkpoint at end of training
        if epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': vars(args)
            }, os.path.join(args.save_dir, 'final_model.pt'))
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()