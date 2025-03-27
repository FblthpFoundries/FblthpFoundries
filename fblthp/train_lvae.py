import os
import argparse
import math
import random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from tqdm import tqdm

# Import our custom modules
from data.datasets import get_dataloaders, MagicCardDataset
from data.mana_vocab import ManaVocabulary

# Import model components
from models.LightweightVAE import LightweightEncoder, LightweightDecoder, PositionalEncoding, LightweightVAE

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set up argument parser
parser = argparse.ArgumentParser(description="Train LightweightVAE for Magic: The Gathering card generation")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent space")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
parser.add_argument("--text_embed_dim", type=int, default=128, help="Text embedding dimension")
parser.add_argument("--mana_embed_dim", type=int, default=16, help="Mana embedding dimension")
parser.add_argument("--kl_weight", type=float, default=0.0001, help="KL divergence weight")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
parser.add_argument("--wandb_project", type=str, default="mtg-vae", help="Wandb project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--prepare_corpus", action="store_true", help="Prepare corpus if not already processed")
parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient clipping threshold")
parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging metrics during training")
parser.add_argument("--log_examples_interval", type=int, default=5, help="Epoch interval for logging examples")

# Parse arguments
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

# Create save directory
os.makedirs(args.save_dir, exist_ok=True)

def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the total gradient norm for the model.
    
    Args:
        model: The model
        
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train_epoch(model: LightweightVAE, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device,
                epoch: int,
                grad_clip: float,
                log_interval: int) -> Dict[str, float]:
    """Train the model for one epoch.
    
    Args:
        model: The VAE model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        grad_clip: Gradient clipping threshold
        log_interval: Interval for logging metrics during training
        
    Returns:
        Dictionary of average loss values
    """
    model.train()
    
    # Track metrics
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_grad_norm = 0.0
    field_losses = {
        'name': 0.0, 'type': 0.0, 
        'oracle': 0.0, 'flavor': 0.0,
        'mana': 0.0, 'cmc': 0.0,
        'power': 0.0, 'toughness': 0.0, 
        'loyalty': 0.0
    }
    
    # Training loop with progress bar
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        reconstructions = model(batch)
        
        # Compute loss
        loss, loss_components = model.compute_loss(batch, reconstructions)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = compute_gradient_norm(model)
        total_grad_norm += grad_norm
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # After loss.backward() but before optimizer.step()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"Gradient NaN in: {name}")
        #             # Print some parameter stats
        #             if param.numel() < 10:
        #                 print(f"Parameter values: {param.data}")
        #             else:
        #                 print(f"Parameter stats: min={param.data.min()}, max={param.data.max()}")
        
        # Optimize
        optimizer.step()
        
        # Update metrics
        batch_size = batch['name_tokens'].size(0)
        total_loss += loss.item() * batch_size
        total_recon_loss += loss_components['recon_loss'].item() * batch_size
        total_kl_loss += loss_components['kl_loss'].item() * batch_size
        
        # Track field-specific losses
        for field in field_losses.keys():
            if f'{field}_loss' in loss_components:
                field_losses[field] += loss_components[f'{field}_loss'].item() * batch_size
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}", 
                         recon=f"{loss_components['recon_loss'].item():.4f}", 
                         kl=f"{loss_components['kl_loss'].item():.4f}",
                         grad=f"{grad_norm:.4f}")
        
        # Log metrics at intervals
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
            # Calculate metrics for this batch
            step = (epoch - 1) * len(dataloader) + batch_idx
            
            # Log to wandb
            wandb.log({
                'batch_loss': loss.item(),
                'batch_recon_loss': loss_components['recon_loss'].item(),
                'batch_kl_loss': loss_components['kl_loss'].item(),
                'gradient_norm': grad_norm,
                'batch': batch_idx,
                'step': step,
            }, step=step)
            
            # Log field-specific losses
            field_metrics = {}
            for field in field_losses.keys():
                if f'{field}_loss' in loss_components:
                    field_metrics[f'batch_{field}_loss'] = loss_components[f'{field}_loss'].item()
            
            if field_metrics:
                wandb.log(field_metrics, step=step)
    
    # Compute average losses
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss = total_loss / num_samples
    avg_recon_loss = total_recon_loss / num_samples
    avg_kl_loss = total_kl_loss / num_samples
    avg_grad_norm = total_grad_norm / num_batches
    
    avg_field_losses = {
        field: loss / num_samples for field, loss in field_losses.items()
    }
    
    # Combine all metrics
    metrics = {
        'train_loss': avg_loss,
        'train_recon_loss': avg_recon_loss,
        'train_kl_loss': avg_kl_loss,
        'train_grad_norm': avg_grad_norm,
    }
    metrics.update({f'train_{field}_loss': loss for field, loss in avg_field_losses.items()})
    
    return metrics

def validate(model: LightweightVAE, 
             dataloader: DataLoader, 
             device: torch.device,
             epoch: int,
             num_examples: int = 5) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Validate the model on the validation set.
    
    Args:
        model: The VAE model
        dataloader: Validation dataloader
        device: Device to use
        epoch: Current epoch number
        num_examples: Number of examples to generate
        
    Returns:
        Tuple of (metrics, examples)
    """
    model.eval()
    
    # Track metrics
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    field_losses = {
        'name': 0.0, 'type': 0.0, 
        'oracle': 0.0, 'flavor': 0.0,
        'mana': 0.0, 'cmc': 0.0,
        'power': 0.0, 'toughness': 0.0, 
        'loyalty': 0.0
    }
    
    # For visualization
    examples = []
    mana_vocab = ManaVocabulary()
    
    with torch.no_grad():
        # Validation loop with progress bar
        pbar = tqdm(dataloader, desc=f"Validate Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            reconstructions = model(batch)
            
            # Compute loss
            loss, loss_components = model.compute_loss(batch, reconstructions)
            
            # Update metrics
            batch_size = batch['name_tokens'].size(0)
            total_loss += loss.item() * batch_size
            total_recon_loss += loss_components['recon_loss'].item() * batch_size
            total_kl_loss += loss_components['kl_loss'].item() * batch_size
            
            # Track field-specific losses
            for field in field_losses.keys():
                if f'{field}_loss' in loss_components:
                    field_losses[field] += loss_components[f'{field}_loss'].item() * batch_size
            
            # Collect examples for visualization (first batch only)
            if batch_idx == 0 and len(examples) < num_examples:
                for i in range(min(num_examples, batch_size)):
                    # Get original and reconstructed tokens
                    example = {
                        'original': {
                            'name': batch['name'][i],
                            'type': batch['type_line'][i],
                            'oracle': batch['oracle_text'][i],
                            'flavor': batch.get('flavor_text', [''])[i] if 'flavor_text' in batch else '',
                            'mana': batch['mc'][i],
                            'cmc': batch['cmc'][i].item(),
                            'power': batch['power'][i].item() if batch['power'][i].item() > 0 else None,
                            'toughness': batch['toughness'][i].item() if batch['toughness'][i].item() > 0 else None
                        },
                        'latent': reconstructions['z'][i].cpu().numpy().tolist()
                    }
                    
                    # Generate a completely new card from the same latent space
                    new_z = reconstructions['z'][i].unsqueeze(0)
                    generated = model.decoder.generate(new_z)
                    
                    # Add generated card to example
                    example['generated'] = {
                        'name_tokens': generated['name_tokens'][0].cpu().numpy().tolist(),
                        'type_tokens': generated['type_tokens'][0].cpu().numpy().tolist(),
                        'oracle_tokens': generated['oracle_tokens'][0].cpu().numpy().tolist(),
                        'flavor_tokens': generated.get('flavor_tokens', [0])[0].cpu().numpy().tolist() 
                                          if 'flavor_tokens' in generated else [],
                        'mana_tokens': generated['mana_tokens'][0].cpu().numpy().tolist(),
                        'cmc': generated['cmc'][0].item(),
                        'power': generated['power'][0].item() if generated['power'][0].item() > 0 else None,
                        'toughness': generated['toughness'][0].item() if generated['toughness'][0].item() > 0 else None
                    }
                    
                    examples.append(example)
            
            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Log validation batch metrics
            step = epoch * len(dataloader) + batch_idx
            if batch_idx == 0 or (batch_idx + 1) % (len(dataloader) // 3) == 0:
                wandb.log({
                    'val_batch_loss': loss.item(),
                    'val_batch_recon_loss': loss_components['recon_loss'].item(),
                    'val_batch_kl_loss': loss_components['kl_loss'].item(),
                    'val_step': step
                }, step=step)
    
    # Compute average losses
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_recon_loss = total_recon_loss / num_samples
    avg_kl_loss = total_kl_loss / num_samples
    
    avg_field_losses = {
        field: loss / num_samples for field, loss in field_losses.items()
    }
    
    # Combine all metrics
    metrics = {
        'val_loss': avg_loss,
        'val_recon_loss': avg_recon_loss,
        'val_kl_loss': avg_kl_loss,
    }
    metrics.update({f'val_{field}_loss': loss for field, loss in avg_field_losses.items()})
    
    return metrics, examples

def generate_random_cards(model: LightweightVAE, 
                          num_samples: int, 
                          device: torch.device,
                          tokenizer=None) -> Dict[str, Any]:
    """Generate random cards by sampling from the latent space.
    
    Args:
        model: The VAE model
        num_samples: Number of samples to generate
        device: Device to use
        tokenizer: Tokenizer for decoding token sequences
        
    Returns:
        Dictionary of generated cards
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from prior
        z = torch.randn(num_samples, model.latent_dim, device=device)
        generated = model.decoder.generate(z)
        
        # Create samples list
        samples = []
        for i in range(num_samples):
            sample = {
                'name_tokens': generated['name_tokens'][i].cpu().numpy().tolist(),
                'type_tokens': generated['type_tokens'][i].cpu().numpy().tolist(),
                'oracle_tokens': generated['oracle_tokens'][i].cpu().numpy().tolist(),
                'flavor_tokens': generated.get('flavor_tokens', [0])[i].cpu().numpy().tolist()
                                  if 'flavor_tokens' in generated else [],
                'mana_tokens': generated['mana_tokens'][i].cpu().numpy().tolist(),
                'cmc': generated['cmc'][i].item(),
                'power': generated['power'][i].item() if generated['power'][i].item() > 0 else None,
                'toughness': generated['toughness'][i].item() if generated['toughness'][i].item() > 0 else None
            }
            samples.append(sample)
    
    return {'random_samples': samples}

def main():
    """Main training function."""
    
    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )
    
    # Log hyperparameters
    config = wandb.config
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataloader, val_dataloader = get_dataloaders(
        test_set_portion=0.1,
        batch_size=args.batch_size,
        seed=args.seed,
        prepare_corpus=args.prepare_corpus
    )
    
    # Get vocabulary sizes from dataset
    sample_batch = next(iter(train_dataloader))
    main_vocab_size = 30000
    name_type_vocab_size = 30000
    mana_vocab_size = 80
    
    print(f"Vocabulary sizes - Main: {main_vocab_size}, Name/Type: {name_type_vocab_size}, Mana: {mana_vocab_size}")
    
    # Create model
    model = LightweightVAE(
        main_vocab_size=main_vocab_size,
        name_type_vocab_size=name_type_vocab_size,
        mana_vocab_size=mana_vocab_size,
        text_embed_dim=args.text_embed_dim,
        mana_embed_dim=args.mana_embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_oracle_len=sample_batch['oracle_tokens'].size(1),
        max_flavor_len=sample_batch['flavor_tokens'].size(1) if 'flavor_tokens' in sample_batch else 50,
        max_name_len=sample_batch['name_tokens'].size(1),
        max_type_len=sample_batch['type_tokens'].size(1),
        max_mana_len=sample_batch['mana_tokens'].size(1),
        kl_weight=args.kl_weight
    ).to(device)
    
    # Log model architecture and parameter count
    wandb.watch(model, log="all", log_freq=100)
    
    # Count and log total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Keep track of best model
    best_val_loss = float('inf')
    
    # Create a custom wandb Table for tracking best examples
    examples_table = wandb.Table(columns=["epoch", "name", "type", "oracle_text", "cmc", "loss"])
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_metrics = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            device, 
            epoch, 
            grad_clip=args.grad_clip,
            log_interval=args.log_interval
        )
        
        # Validate
        val_metrics, examples = validate(model, val_dataloader, device, epoch)
        
        # Generate random samples
        random_samples = generate_random_cards(model, num_samples=5, device=device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        combined_metrics = {**train_metrics, **val_metrics, 'learning_rate': current_lr, 'epoch': epoch}
        wandb.log(combined_metrics)
        
        # Track performance over time with a custom plot
        wandb.log({
            "loss_comparison": wandb.plot.line_series(
                xs=list(range(1, epoch + 1)),
                ys=[
                    [m.get('train_loss', 0) for m in [train_metrics]],
                    [m.get('val_loss', 0) for m in [val_metrics]]
                ],
                keys=["train", "validation"],
                title="Loss Progression",
                xname="epoch"
            )
        })
        
        # Add latent space visualization (2D projection of examples)
        if examples and len(examples) > 0:
            # Extract latent vectors for visualization
            latent_vecs = [ex['latent'] for ex in examples]
            if latent_vecs and len(latent_vecs) > 0:
                try:
                    # Create a scatter plot for latent space visualization
                    latent_data = [[i] + vec for i, vec in enumerate(latent_vecs)]
                    latent_table = wandb.Table(data=latent_data, columns=["index"] + [f"dim_{i}" for i in range(len(latent_vecs[0]))])
                    wandb.log({"latent_space": wandb.plot.scatter(latent_table, "dim_0", "dim_1", "index")})
                except Exception as e:
                    print(f"Error creating latent space visualization: {e}")
        
        # Log examples (using the specified interval)
        should_log_examples = (epoch % args.log_examples_interval == 0 or epoch == 1 or epoch == args.epochs)
        if should_log_examples:
            wandb.log({
                "examples": examples,
                "random_samples": random_samples["random_samples"]
            })
            
            # Add best examples to the tracking table
            if examples and len(examples) > 0:
                for i, ex in enumerate(examples[:3]):  # Just add top 3 examples
                    try:
                        examples_table.add_data(
                            epoch,
                            ex['original']['name'],
                            ex['original']['type'],
                            ex['original']['oracle'],
                            ex['original']['cmc'],
                            val_metrics['val_loss']
                        )
                    except Exception as e:
                        print(f"Error adding to examples table: {e}")
        
        # Print metrics
        print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Gradient Norm: {train_metrics['train_grad_norm']:.4f}, "
              f"KL Weight: {model.kl_weight:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': vars(args)
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"Saved best model with validation loss {best_val_loss:.4f}")
            
            # Log this as a special event
            wandb.log({"best_model_updated": epoch, "best_val_loss": best_val_loss})
        
        # Save checkpoint periodically
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'config': vars(args)
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Log the examples table at the end of training
    wandb.log({"training_examples_history": examples_table})
    
    # Final model save
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['val_loss'],
        'config': vars(args)
    }, os.path.join(args.save_dir, 'final_model.pt'))
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()