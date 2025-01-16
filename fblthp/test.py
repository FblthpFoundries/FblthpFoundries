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
    message=r"You are using `torch.load` with `weights_only=False`"
)



from models.TransformerVAE import TransformerVAE
import torch
import os
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split
from data.datasets import get_dataloaders
from tabulate import tabulate
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
import Levenshtein
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import ast
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_test_func(model):
    return {"loss": -1}


def recon_loss(model, test_dataloader):
    l_recon = 0.0
    for test_x in test_dataloader:
        test_x = test_x["ids"].to(device)
        test_logits, test_mu, test_logvar = model(test_x, target_seq=test_x)
        total, recon, scaled, kl = model.vae_loss(test_logits, test_x, test_mu, test_logvar, kl_weight=0.03, free_bits=0.2)
        l_recon += recon.item()
    l_recon /= len(test_dataloader)
    return {"loss": l_recon}

def attribute_reconstruction_loss(model, test_dataloader, tokenizer, n=32):
    def yoink(text, attribute):
        pattern = fr"<{attribute}>(.*?)<\\{attribute}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().replace("}", "").replace("{", "").replace("*", "0")
        return "0"
    def mop(text, make_zero=False):
        if isinstance(text, str):
            text = text.replace("}", "").replace("{", "").replace(" ", "").replace("*", "0")
            if text == "nan" or text == "<empty>":
                return "0"
            if make_zero and text == "":
                return "0"
            if "+" in text:
                text = str(eval(text))
            return text
        if math.isnan(text):
            return "0"
        return text
    model.eval()  # Set model to evaluation mode

    intermediates = defaultdict(list)


    i = 0
    pbar = tqdm(total=n, desc="Attribute Reconstruction Loss")
    while i < n:
        x = next(iter(test_dataloader))
        

        x_i = x["ids"].to(device)
        num_examples = x_i.shape[0]
        mu, logvar = model.encoder(x_i)
        z = model.reparameterize(mu, logvar)
        reconstructed_token_ids = model.generate(z, max_len=125)
        original_texts = x["originals"]
        for example in range(num_examples):
            reconstructed_text = tokenizer.decode(reconstructed_token_ids[example].squeeze().tolist(), skip_special_tokens=False)
            original_text = original_texts[example]
            #Mana cost

            left = mop(x["mc"][example])
            right = mop(yoink(reconstructed_text, "mc"))
            left_cmc = sum([int(x) if x.isdigit() else {"X": 0, "Y": 0, "W": 1, "U": 1, "B": 1 , "R": 1, "G": 1, "C": 1, "P": 1, "/": -1, }[x] for x in left])
            right_cmc = sum([int(x) if x.isdigit() else {"X": 0, "Y": 0, "W": 1, "U": 1, "B": 1 , "R": 1, "G": 1, "C": 1, "P": 1, "/": -1, }[x] for x in right])

            intermediates["MC Levenshtein"].append(Levenshtein.ratio(left, right))
            intermediates["CMC MSE"].append((left_cmc - right_cmc)**2)

            #Power

            left = mop(x["power"][example], make_zero=True)
            right = mop(yoink(reconstructed_text, "power"), make_zero=True)
            intermediates["Power MSE"].append((int(left) - int(right))**2)

            #Toughness

            left = mop(x["toughness"][example], make_zero=True)
            right = mop(yoink(reconstructed_text, "toughness"), make_zero=True)
            intermediates["Toughness MSE"].append((int(left) - int(right))**2)

            i += 1
            pbar.update(1)
            if i >= n:
                break

    pbar.close()
    returned = {}

    for attribute in intermediates:
        returned[attribute] = np.mean(intermediates[attribute])

    return returned


def tsne_graphs(model, test_dataloader, tokenizer, n=32):
    
    def yoink(text, attribute):
        pattern = fr"<{attribute}>(.*?)<\\{attribute}>"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return None
    def rgb_to_hex(r, g, b):
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    colors_dict = {
        "W": (255, 255, 240),
        "U": (0, 115, 230),
        "B": (15, 15, 15),
        "R": (200, 0, 0),
        "G": (0, 150, 75),
        "C": (192, 192, 192),
    }

    pbar = tqdm(total=n, desc="t-SNE Graphs")
    bigboy = torch.zeros((n, model.encoder.fc_mu.out_features), device=device)
    colors = []
    cmcs = []
    i = 0
    it = iter(test_dataloader)
    while i < n:
        x = next(it)
        x_i = x["ids"].to(device)

        mu, logvar = model.encoder(x_i)
        z = model.reparameterize(mu, logvar)

        needed = min(n - i, x_i.shape[0])

        bigboy[i:i+needed, :] = z[:needed, :]

        for j in range(x_i.shape[0]):
            meed = []
            mc = x["mc"][j]
            if not isinstance(mc, str) and math.isnan(mc):
                colors.append(rgb_to_hex(0, 255, 255))
            else:
                mc = mc.replace(" ", "").replace("{", "").replace("}", "")
                for chr in mc:
                    if chr.upper() in colors_dict:
                        meed.append(colors_dict[chr.upper()])
                if len(meed) == 0:
                    meed.append(colors_dict["C"])
                mean_color = np.mean(meed, axis=0).astype(int)
                hex = rgb_to_hex(*mean_color)
                colors.append(hex)

            cmcs.append(x["cmc"][j])
            i += 1
            pbar.update(1)
            if i >= n:
                break

    print("Starting plot")
    print(colors[:10])
    tsne = TSNE(n_components=2, random_state=42)
    z_embedded = tsne.fit_transform(bigboy.cpu().detach().numpy())
    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=colors)
    plt.title("t-SNE of Latent Space")
    plt.show()
    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=cmcs, cmap='viridis')
    plt.title("t-SNE of Latent Space")
    plt.show()


    return {}

# Function to generate and log a card to wandb
def log_generated_card(model, tokenizer, device='cuda', n=1):
    model.eval()  # Set model to evaluation mode
    for i in range(n):
        # Generate a random latent vector (or sample from the latent space)
        latent_vector = torch.randn(1, model.encoder.fc_mu.out_features).to(device)
        
        # Generate a card from the latent vector
        generated_token_ids = model.generate(latent_vector, max_len=125)[0]  # Adjust max_len if needed
        #print(generated_token_ids)

        if isinstance(generated_token_ids, torch.Tensor):
            if generated_token_ids.ndim == 0:  # Handle scalar tensor
                generated_token_ids = [generated_token_ids.item()]
            else:
                generated_token_ids = generated_token_ids.squeeze().tolist()

        # Convert token IDs back to human-readable text
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        
        # Log the generated card text to wandb
        #wandb.log({"generated_card": generated_text})
        print("ids[:10]:", generated_token_ids[:10])
        print("decoded text: ", generated_text.replace('<pad>', '').replace('<eos>', ''))

    model.train()  # Switch back to training mode
    return {}

def reconstruct_cards(model, tokenizer, dataloader, device='cuda', n=1):
    model.eval()  # Set model to evaluation mode
    reconstructed_texts = []
    original_texts = []
    
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            if i >= n:
                break
            x = x["ids"].to(device)
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


def diagnostic_test(model, test_dataloader, tokenizer, hypers, beta):
    print("-Triggering DIAGNOSTIC LOGGING! -_-")
    if test_dataloader is None:
        print("No test data!")
        return False
    model.eval()
    with torch.no_grad():
        l_total = 0.0
        l_recon = 0.0
        l_scaled = 0.0
        l_kl = 0.0
        for test_x in test_dataloader:
            test_x = test_x["ids"].to(device)
            test_logits, test_mu, test_logvar = model(test_x, target_seq=test_x)
            total, recon, kl, scaled  = model.vae_loss(test_logits, test_x, test_mu, test_logvar, kl_weight=beta, free_bits=hypers["free_bits"])
            l_total += total.item()
            l_recon += recon.item()
            l_scaled += scaled.item()
            l_kl += kl.item()
        l_total /= len(test_dataloader)
        l_recon /= len(test_dataloader)
        l_scaled /= len(test_dataloader)
        l_kl /= len(test_dataloader)
        # Log the validation loss to wandb
        metrics = {
            "val_loss": l_total,
            "val_ce_loss": l_recon,
            "val_kl_loss": l_kl,
            "val_scaled_kl_loss": l_scaled,
        }
    model.train()
    print("----------------[Validation Loss:]-----------------")
    print(f"Total loss: {l_total:.4f}, CE loss: {l_recon:.4f}, KL loss: {l_kl:.4f}, Scaled KL loss: {l_scaled:.4f}")
    print("----------[Sample cards:]----------------- ")
    log_generated_card(model, tokenizer, device, n=10)
    print("----------[Reconstructed cards:]----------------- ")
    reconstruct_cards(model, tokenizer, test_dataloader, device, n=10)
    return metrics

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODELS_DIR = "weights_stash"

    tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")

    seed = 42
    test_set_portion = 0.05
    batch_size = 32

    _, test_dataloader = get_dataloaders(test_set_portion, seed, batch_size)

    models = {
        #"512d-best": "12-3-best.pt",
        #"512d-overfit": "12-3-overfit.pt",
        #"768d-best": "12-5-best.pt",
        #"768d-overfit": "12-5-overfit.pt",
        #"1024d-best": "12-5-1024dim-best.pt",
        #"1024d-4l-encoder-best": "12-6-4layerencoder-best.pt",
        #"micro-best": "micro-best.pt",
        #"micro-overfit": "micro-overfit.pt",
        #"adamant-will": "adamant-will_best_checkpoint.pt",
        #"big-dropout": "big-dropout.pt",
        "ugins-conjurant": "ugins-conjurant_best_checkpoint.pt",
        "battlefield-promotion":"battlefield-promotion_best_checkpoint.pt"
        }
    
    return_test_fns = [
        ("Test Function", eval_test_func, {}),
        ("CE Reconstruction Loss", recon_loss, {"test_dataloader": test_dataloader}),
        ("Attribute Reconstruction Loss", attribute_reconstruction_loss, {"test_dataloader": test_dataloader, "tokenizer": tokenizer, "n": 1300}),
        ("TSNE Graphs", tsne_graphs, {"test_dataloader": test_dataloader, "tokenizer": tokenizer, "n": 500}),
        ("Diagnostic Test", diagnostic_test, {"test_dataloader": test_dataloader, "tokenizer": tokenizer, "hypers": {"free_bits": 0.2}, "beta": 0.01}),
        ("Log Cards", log_generated_card, {"tokenizer": tokenizer, "n": 1}),
    ]

    results = {name: [] for name, _, _ in return_test_fns}


    for model_name in models:
        print(f"Testing model {model_name}")
        #Load model checkpoint
        checkpoint = torch.load(os.path.join(MODELS_DIR, models[model_name]))
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

        test_set_portion = checkpoint["hypers"]["test_set_portion"]
        seed = checkpoint["hypers"]["seed"]

        #Load model into memory and set to eval mode
        model = TransformerVAE(vocab_size, embed_dim, num_heads, hidden_dim, num_encoder_layers, num_decoder_layers, max_len, dropout=dropout).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        for name, test_func, args in return_test_fns:
            

            result = test_func(model, **args)

            #print(result)
            results[name].append((model_name, result))
            

    # Collect all results into a list of rows for the table


    for result_name in results:
        table_data = []
        print(f"Results for {result_name}")
        
        # Prepare table headers dynamically based on dictionary keys
        headers = ["Model Name"]  # Start with a column for model names
        first_result = None
        for model_name, result in results[result_name]:
            if first_result is None:
                first_result = result  # Get the first result to extract column names
            # Add model name to table data
            row = [model_name] + [result[key] for key in first_result]
            table_data.append(row)

        # Include keys of the dictionary as headers
        if first_result is not None:
            headers += list(first_result.keys())

        # Print the table using tabulate
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        print("\n")