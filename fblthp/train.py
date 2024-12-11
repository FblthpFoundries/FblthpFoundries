import warnings
# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Using `TRANSFORMERS_CACHE` is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch.cuda.amp.GradScaler*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch.cuda.amp.autocast*"
)

from datetime import datetime
import torch
from models.checkpoints import save_state, create_gods
from models.loss import compute_beta
from data.datasets import get_dataloaders
from data.tokenizers import get_mtg_tokenizer
from test import diagnostic_test
from utils.logs import init_wandb, log_memory, log_batch, log_eval
import wandb
from tqdm import tqdm  # Import tqdm for progress bars
import traceback
from torch.cuda.amp import autocast, GradScaler

import argparse
from config import HYPERS

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_vae(model, optimizer, scheduler, tokenizer, hypers, start_step, train_dataloader, 
              test_dataloader=None
        ):
    model.train()
    scaler = GradScaler()
    
    best_loss = 1000000
    print(f"Training model {hypers['name']}")
    # Initialize a single tqdm progress bar for the entire training process
    with tqdm(total=hypers["num_steps"], initial=start_step, unit="step") as pbar:
        step = start_step
        accumulation_counter = 0
        t_loss = 0
        t_ce_loss = 0
        t_kl_loss = 0
        t_scaled_kld_loss = 0
        data_iter = iter(train_dataloader)  # Create a single iterator for the dataloader
        try:
            while step < hypers["num_steps"]:
                try:
                    x_i = next(data_iter)  # Get the next batch
                except StopIteration:
                    # Restart the dataloader when it runs out of batches
                    data_iter = iter(train_dataloader)
                    x_i = next(data_iter)

                x = x_i["ids"].to(device)
                # Compute the teacher forcing ratio

                

                # Forward pass through the VAE
                with autocast():
                    logits, mu, logvar = model(x, target_seq=x)
                    beta = compute_beta(step, hypers["beta_start"], hypers["beta_end"], hypers["beta_warmup_steps"])
                    loss, ce_loss, kl_loss, scaled_kld_loss = model.vae_loss(logits, x, mu, logvar, kl_weight=beta, free_bits=hypers["free_bits"])
                    loss = loss / hypers["accumulation_steps"]
                    ce_loss = ce_loss / hypers["accumulation_steps"]
                    kl_loss = kl_loss / hypers["accumulation_steps"]
                    scaled_kld_loss = scaled_kld_loss / hypers["accumulation_steps"]
                    t_loss += loss.item()
                    t_ce_loss += ce_loss.item()
                    t_kl_loss += kl_loss.item()
                    t_scaled_kld_loss += scaled_kld_loss.item()

                scaler.scale(loss).backward()

                accumulation_counter += 1
                if (accumulation_counter) % hypers["accumulation_steps"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    # Step the learning rate scheduler
                    scheduler.step()

                    #wandb log
                    log_batch(t_loss, step, scheduler.get_last_lr()[0],
                            t_ce_loss, t_kl_loss, t_scaled_kld_loss, beta)
                    
                    t_loss = 0
                    t_ce_loss = 0
                    t_kl_loss = 0
                    t_scaled_kld_loss = 0
                    # Update tqdm with the current step and loss
                    step += 1  # Increment step counter
                    pbar.set_postfix(ce_loss=ce_loss.item() * hypers["accumulation_steps"])
                    pbar.update(1)

                    # Log additional information at intervals
                    if step % 2000 == 0:
                        metrics = diagnostic_test(model, test_dataloader, tokenizer, hypers, beta)
                        log_eval(metrics)

                        exceptional_performance = metrics["val_ce_loss"] < best_loss

                        if exceptional_performance:
                            print("Exceptional performance detected. Saving model...")
                            best_loss = metrics["val_ce_loss"]
                            save_state(model, optimizer, scheduler, step, f"{hypers['name']}_best_checkpoint.pt", hypers=hypers)
                        
                        log_memory()
                
                    if step % 10000 == 0:
                        print("Saving checkpoint...")
                        save_state(model, optimizer, scheduler, step, f"{hypers['name']}_checkpoint_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt", hypers=hypers)

                

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            filename = f"{hypers['name']}_canceled_checkpoint_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt"
            save_state(model, optimizer, scheduler, step, filename, hypers=hypers)
            print(f"Checkpoint saved successfully to {filename}. Exiting.")
            return

        print("Training complete!")
        filename = f"{hypers['name']}_final_checkpoint_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt"
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

    # Initialize or load tokenizer
    tokenizer = get_mtg_tokenizer("wordpiece_tokenizer.json")


    model, optimizer, scheduler, step, hypers = create_gods(HYPERS, args.resume_weights)

    # Initialize wandb
    init_wandb(hypers)
    
    model.print_model_size()

    # Load the dataset
    train_dataloader, test_dataloader = get_dataloaders(HYPERS['test_set_portion'], HYPERS['seed'], HYPERS['batch_size']//HYPERS['accumulation_steps'])
    

    # Train the model
    try:
        train_vae(
            model, optimizer, scheduler, tokenizer, hypers, 
            start_step=step, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()
if __name__ == "__main__":
    main()
