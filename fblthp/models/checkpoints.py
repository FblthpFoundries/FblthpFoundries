import torch
import os
from .TransformerVAE import TransformerVAE
from .ADVAE import ADVAE
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import uuid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_state(filename):
    checkpoint = torch.load(filename)

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
    lr_warmup_steps = checkpoint["hypers"]["lr_warmup_steps"]
    beta_start = checkpoint["hypers"]["beta_start"]
    beta_end = checkpoint["hypers"]["beta_end"]
    beta_warmup_steps = checkpoint["hypers"]["beta_warmup_steps"]
    free_bits = checkpoint["hypers"]["free_bits"]
    test_set_portion = checkpoint["hypers"]["test_set_portion"]
    seed = checkpoint["hypers"]["seed"]
    step = checkpoint["step"]
    if "name" in checkpoint["hypers"]:
        name = checkpoint["hypers"]["name"]
    else:
        name = "UnnamedModel"


    model = TransformerVAE(vocab_size, embed_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len, dropout=dropout).to(device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=num_steps
    )
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    hypers = checkpoint["hypers"]


    return model, optimizer, scheduler, step, hypers


def create_gods(hypers, filename=None):
    if filename:
        model, optimizer, scheduler, step, hypers = load_state(filename)
        return model, optimizer, scheduler, step, hypers
    

    if "model" not in hypers:
        hypers["model"] = "TransformerVAE"


    if hypers["model"] == "TransformerVAE":
        model = TransformerVAE(
            vocab_size=hypers["vocab_size"],
            embed_dim=hypers["embed_dim"],
            num_heads=hypers["num_heads"],
            hidden_dim=hypers["hidden_dim"],
            num_encoder_layers=hypers["num_encoder_layers"],
            num_decoder_layers=hypers["num_decoder_layers"],
            max_len=hypers["max_len"],
            dropout=hypers["dropout"],
        ).to(device)
        hypers["id"] = str(uuid.uuid4())
    elif hypers["model"] == "ADVAE":
        model = ADVAE(
            vocab_size=hypers["vocab_size"],
            embed_dim=hypers["embed_dim"],
            latent_dim=hypers["latent_dim"],
            num_heads=hypers["num_heads"],
            hidden_dim=hypers["hidden_dim"],
            num_encoder_layers=hypers["num_encoder_layers"],
            num_decoder_layers=hypers["num_decoder_layers"],
            max_len=hypers["max_len"],
            dropout=hypers["dropout"],
        ).to(device)
        hypers["id"] = str(uuid.uuid4())
    else:
        raise ValueError(f"Model {hypers['model']} not supported.")
    
    optimizer = optim.Adam(model.parameters(), lr=hypers["learning_rate"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=hypers["lr_warmup_steps"], num_training_steps=hypers["num_steps"]
    )
    step = 1

    return model, optimizer, scheduler, step, hypers