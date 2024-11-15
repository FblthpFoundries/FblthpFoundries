import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from collections import defaultdict

import re

# Assuming your TransformerVAE model and data loader are set up like this
# model: the TransformerVAE model
# dataloader: data loader providing batches of cards
def _parse_card_data(self, input_text):
    cards = []
    
    # Define patterns for each part of the card
    card_pattern = r'<tl>(.*?)<\\tl>'
    name_pattern = r'<name>(.*?)<\\name>'
    mc_pattern = r'<mc>(.*?)<\\mc>'
    ot_pattern = r'<ot>(.*?)<\\ot>'
    power_pattern = r'<power>(.*?)<\\power>'
    toughness_pattern = r'<toughness>(.*?)<\\toughness>'
    loyalty_pattern = r'<loyalty>(.*?)<\\loyalty>'
    ft_pattern = r'<ft>(.*?)<\\ft>'
    
    # Split the input into sections for each card
    card_matches = re.findall(r'<tl>.*?(?=<tl>|$)', input_text, re.DOTALL)
    
    for card_match in card_matches:
        card = {}
        
        # Extract each component using the patterns
        if not re.search(card_pattern, card_match):
            continue
        card['type_line'] = re.search(card_pattern, card_match).group(1).strip()
        
        name = re.search(name_pattern, card_match)
        card['name'] = name.group(1).strip() if name else None
        
        mc = re.search(mc_pattern, card_match)
        card['mana_cost'] = mc.group(1).strip() if mc else None
        
        ot = re.search(ot_pattern, card_match)
        card['oracle_text'] = re.sub(r'<nl>', '\n', ot.group(1).strip()) if ot else None
        if not card['oracle_text'] :
            continue
        card['oracle_text'] = card['oracle_text'].replace('<br>', '\n')
        if not card['name']:
            continue
        card['oracle_text'] = card['oracle_text'].replace('~', card['name'])
        
        power = re.search(power_pattern, card_match)
        card['power'] = power.group(1).strip() if power else None
        
        toughness = re.search(toughness_pattern, card_match)
        card['toughness'] = toughness.group(1).strip() if toughness else None
        
        loyalty = re.search(loyalty_pattern, card_match)
        card['loyalty'] = loyalty.group(1).strip() if loyalty else None
        
        ft = re.search(ft_pattern, card_match)
        card['flavor_text'] = re.sub(r'<nl>', '\n', ft.group(1).strip()) if ft else None
        
        cards.append(card)
    
    return cards
def get_color(mana_cost):
    color = []
    if 'W' in mana_cost:
        color.append('W')
    if 'U' in mana_cost:
        color.append('U')
    if 'B' in mana_cost:
        color.append('B')
    if 'R' in mana_cost:
        color.append('R')
    if 'G' in mana_cost:
        color.append('G')
    if len(color) == 0:
        color.append('C')
    return color
def get_cmc(mana_cost):
    cmc = 0
    for c in mana_cost:
        if c.isnumeric():
            cmc += int(c)
        elif c in ['W', 'U', 'B', 'R', 'G', 'C']:
            cmc += 1
    return cmc
def visualize_latent_space(model, dataloader, n_samples=1000, perplexity=30, learning_rate=200):
    """
    Visualize the latent space using t-SNE.
    
    Parameters:
    - model: Your trained TransformerVAE model.
    - dataloader: Dataloader providing batches of input cards.
    - n_samples: Number of cards to sample for the t-SNE plot.
    - perplexity: t-SNE perplexity (try values around 30-50).
    - learning_rate: t-SNE learning rate (200 is usually a good start).
    """
    model.eval()
    latent_vectors = []
    labels = defaultdict(list)
    cats = ["color", "mana_cost", "power", "toughness", "loyalty"]
    
    with torch.no_grad():
        for i, card in enumerate(dataloader):
            if i * dataloader.batch_size >= n_samples:
                break
            # Encode to get the latent representation
            
            labels["power"].append(parsed["power"] if parsed["power"] else -1)
            labels["toughness"].append(parsed["toughness"] if parsed["toughness"] else -1)
            labels["loyalty"].append(parsed["loyalty"] if parsed["loyalty"] else -1)
            labels["color"].append(get_color(parsed["mana_cost"]))
            labels["cmc"].append(get_cmc(parsed["mana_cost"]))



            latent = model.encode(card.to(model.device))  # Change to your model's encode method
            latent_vectors.append(latent.cpu().numpy())
            
            for cat in cats:
                if cat in 

            labels.extend(label.numpy())  # Optional: change label to card attributes if needed

    # Stack all latent vectors for t-SNE
    latent_vectors = np.concatenate(latent_vectors, axis=0)

    # Run t-SNE on the latent vectors
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    tsne_results = tsne.fit_transform(latent_vectors)

    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Card Attributes')  # Only if labels are provided
    plt.title("t-SNE of TransformerVAE Latent Space")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()



if __name__ == '__main__':

    if not os.path.exists("wordpiece_tokenizer.json"):
        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        tokenizer.enable_padding(pad_id=0, pad_token=special_tokens[0], length=max_len)
        tokenizer.enable_truncation(max_length=max_len)
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        dataset = MagicCardDataset(tokenizer, max_len=max_len) # just here in order to make corpus.csv on a first go (bad code but it works)
        tokenizer.train(['corpus.csv'], trainer)
        tokenizer.save("wordpiece_tokenizer.json")
    else:
        tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")

    wandb.init(project="TransformerVAE", config={
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "embed_dim": embed_dim,
    "num_heads": num_heads,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "max_len": max_len,
    })
    # Create the VAE model
    vae_model = TransformerVAE(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len)
    vae_model.print_model_size()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae_model = vae_model.to(device)
    
    
    
    # DataLoader (use a dummy dataset for now)
    dataset = MagicCardDataset(tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Optimizer
    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=(num_epochs * len(dataset) // batch_size))

    # Start training
    try:
        train_vae(vae_model, dataloader, optimizer, tokenizer, num_epochs=num_epochs, device=device, scheduler=scheduler, max_len=max_len, decay_rate=decay_rate)
    except Exception as e:
        print(e)
        traceback.print_exc()
    torch.save(vae_model.state_dict(), 'vae_model2.pt')