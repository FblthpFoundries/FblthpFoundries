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
from TransformerVAE import TransformerVAE
import torch
import os
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from train import MagicCardDataset
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

class LabeledMagicCardDataset(Dataset):
    def __init__(self, max_len=125, target="corpus.csv"):
        self.max_len = max_len
        self.corpus_dataframe = pd.read_csv(target)

    def __len__(self):
        return len(self.corpus_dataframe)

    def __getitem__(self, idx):
        row = self.corpus_dataframe.iloc[idx]
        tolist = row.tolist()
        return tolist




def eval_test_func(model):
    return {"loss": -1}


def recon_loss(model, test_dataloader):
    l_recon = 0.0
    for test_x in test_dataloader:
        test_x = test_x.to(device)
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
            return match.group(1).strip()
        return None

    model.eval()  # Set model to evaluation mode

    intermediates = defaultdict(list)


    i = 0
    pbar = tqdm(total=n, desc="Attribute Reconstruction Loss")
    while i < n:
        x = next(iter(test_dataloader))
        num_examples = x.shape[0]

        x = x.to(device)
        mu, logvar = model.encoder(x)
        z = model.reparameterize(mu, logvar)
        reconstructed_token_ids = model.generate(z, max_len=125)

        for example in range(num_examples):
            reconstructed_text = tokenizer.decode(reconstructed_token_ids[example].squeeze().tolist(), skip_special_tokens=False)
            original_text = tokenizer.decode(x[example].squeeze().tolist(), skip_special_tokens=False)
            #Mana cost

            
            left = yoink(original_text, "mc")
            if not left:
                left = "0"
            else:
                left = left.replace("}", "").replace("{", "")
            right = yoink(reconstructed_text, "mc")
            
            if not right:
                right = "0"
            else:
                right = right.replace("}", "").replace("{", "")
            left_cmc = sum([int(x) if x.isdigit() else 1 for x in left.replace(" ", "")])
            right_cmc = sum([int(x) if x.isdigit() else 1 for x in right.replace(" ", "")])

            intermediates["MC Levenshtein"].append(Levenshtein.ratio(left, right))
            intermediates["CMC MSE"].append((left_cmc - right_cmc)**2)

            #Power

            left = yoink(original_text, "power")
            if not left:
                left = "0"
            else:
                left = left.replace("}", "").replace("{", "").replace("*", "0")
            right = yoink(reconstructed_text, "power")
            if not right:
                right = "0"
            else:
                right = right.replace("}", "").replace("{", "").replace("*", "0")
            intermediates["Power MSE"].append((int(left) - int(right))**2)

            #Toughness

            left = yoink(original_text, "toughness")
            if not left:
                left = "0"
            else:
                left = left.replace("}", "").replace("{", "").replace("*", "0")
            right = yoink(reconstructed_text, "toughness")
            if not right:
                right = "0"
            else:
                right = right.replace("}", "").replace("{", "").replace("*", "0")
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
    while i < n:
        x = next(iter(test_dataloader))
        x_i = x["ids"]

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    MODELS_DIR = "checkpoints"
    models = {
        "512d-best": "12-3-best.pt",
        "512d-overfit": "12-3-overfit.pt",
        "768d-best": "12-5-best.pt",
        "768d-overfit": "12-5-overfit.pt",
        "1024d-best": "12-5-1024dim-best.pt",
        "1024d-4l-encoder-best": "12-6-4layerencoder-best.pt",
        }




    tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")

    seed = 42
    test_set_portion = 0.05
    batch_size = 32

    # DataLoader
    dataset = LabeledMagicCardDataset(max_len=125, target="labeled.csv")

    # Get the training dataset
    train_size = int(len(dataset) * (1 - test_set_portion))
    test_size = len(dataset) - train_size
    torch.manual_seed(seed) # Seed so the random split is the same and doesn't contaminate when reloading a model
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True, num_workers=2, persistent_workers=True)
    
    def collate_fn(batch):
        ids = [ast.literal_eval(x[1]) for x in batch]
        ids = torch.tensor(ids, device=device)
        originals = [x[0] for x in batch]
        mc = [x[2] for x in batch]
        power = [x[3] for x in batch]
        toughness = [x[4] for x in batch]
        cmc = [x[5] for x in batch]

        return {
            "ids": ids, 
            "originals": originals,
            "mc": mc,
            "power": power,
            "toughness": toughness,
            "cmc": cmc,
            }
    return_test_fns = [
        #("Test Function", eval_test_func, {}),
        #("CE Reconstruction Loss", recon_loss, {"test_dataloader": test_dataloader}),
        #("Attribute Reconstruction Loss", attribute_reconstruction_loss, {"test_dataloader": test_dataloader, "tokenizer": tokenizer, "n": 1300})
        ("TSNE Graphs", tsne_graphs, {"test_dataloader": DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn), "tokenizer": tokenizer, "n": 500})
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