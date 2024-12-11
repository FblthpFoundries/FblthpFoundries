import wandb
import torch
from torch.cuda import memory_allocated, memory_reserved
def init_wandb(hypers):
    wandb.init(project="Fblthp Foundries",
               name=hypers["name"],
               id=hypers["id"],
               resume="allow",
               config={
                    "learning_rate": hypers["learning_rate"],
                    "steps": hypers["num_steps"],
                    "batch_size": hypers["batch_size"],
                    "embed_dim": hypers["embed_dim"],
                    "num_heads": hypers["num_heads"],
                    "hidden_dim": hypers["hidden_dim"],
                    "beta_start": hypers["beta_start"],
                    "beta_end": hypers["beta_end"],
                    "beta_warmup_steps": hypers["beta_warmup_steps"],
                    "num_encoder_layers": hypers["num_encoder_layers"],
                    "num_decoder_layers": hypers["num_decoder_layers"],
                    "max_len": hypers["max_len"],
                    "dropout": hypers["dropout"],}
                )
def log_memory():
    allocated = memory_allocated() / (1024 ** 2)  # Convert to MB
    reserved = memory_reserved() / (1024 ** 2)    # Convert to MB
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

def log_batch(loss, step, lr, ce_loss, kl_loss, scaled_kld_loss, beta):
    # Log stats to wandb for this batch
    wandb.log({
        "loss": loss,
        "step": step,
        "lr": lr,  # Log the learning rate
        "ce_loss": ce_loss,
        "kl_loss": kl_loss,
        "scaled_kl_loss": scaled_kld_loss,
        "beta": beta,
    })

def log_eval(metrics):
    wandb.log(metrics)