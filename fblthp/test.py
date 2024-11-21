from tokenizers import Tokenizer
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from collections import defaultdict

import re
import argparse

from TransformerVAE import PositionalEncoding, TransformerVAE
from train import MagicCardDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def test_cards(n, vae_model, tokenizer):
    for i in range(n):
        # Generate a random latent vector (or sample from the latent space)
        latent_vector = torch.randn(1, vae_model.encoder.fc_mu.out_features).to(device)
        # Generate a card from the latent vector
        generated_token_ids = vae_model.generate(latent_vector, max_len=50)[0]  # Adjust max_len if needed
        # Convert token IDs back to human-readable text (you need your tokenizer for this)
        generated_text = tokenizer.decode(generated_token_ids.squeeze().tolist(), skip_special_tokens=False)
        
        # Log the generated card text to wandb
        print("ids[:20]:", generated_token_ids.squeeze().tolist()[:20])
        print("decoded text: ", generated_text.replace('<pad>', '').replace('<eos>', ''))

if __name__ == '__main__':
    max_len = 125

    tokenizer = Tokenizer.from_file("wordpiece_tokenizer.json")
    dataset = MagicCardDataset(tokenizer, max_len=max_len) # just here in order to make corpus.csv on a first go (bad code but it works)


    # Argument parser
    parser = argparse.ArgumentParser(description='Load a TransformerVAE model from a file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file.')

    args = parser.parse_args()

    # Load the model
    checkpoint = torch.load(args.model_path)
    vae_model = TransformerVAE(vocab_size=20000, embed_dim=768, \
                               num_heads=12, hidden_dim=3072, num_layers=12, max_len=125)
    
    vae_model.load_state_dict(checkpoint["model_state"])
    vae_model.encoder.pos_encoder = PositionalEncoding(768, max_len*2)
    vae_model.decoder.pos_encoder = PositionalEncoding(768, max_len*2)
    vae_model.print_model_size()
    vae_model = vae_model.to(device)
    
    
    # DataLoader (use a dummy dataset for now)
    dataset = MagicCardDataset(tokenizer, max_len=max_len)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    vae_model.eval()  # Set model to evaluation mode

    test_card = tokenizer.encode("<tl>Instant<\\tl><name>Lightning Bolt<\\name><mc>{R}<\\mc><ot>Lightning Bolt deals 3 damage to any target.<\\ot><eos>").ids
    sos_token_id = tokenizer.token_to_id('<sos>') if '<sos>' in tokenizer.get_vocab() else 1
    print(f"sos_token_id: {sos_token_id}")

    test_card = torch.tensor([sos_token_id] + test_card, device=device).unsqueeze(0).to(device)
    print(test_card)

    #test_card = test_card[:,:max_len]
    encoded = vae_model.encoder(test_card)

    print(encoded[0].shape, encoded[1].shape)

    z = vae_model.reparameterize(encoded[0], encoded[1])

    print(z.shape)


    out = vae_model.decoder(z, max_len=125)[1]

    decoded_x = out.float()
    left = decoded_x.view(-1, decoded_x.size(-1))
    right = test_card.reshape(-1).to(device)[1:]
    print(left.shape, right.shape)
    recon_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<pad>'))(left, right)
    print(recon_loss)


    #simulate decoder

    # # Initialize the output sequence with the <sos> token
    # generated_sequence = torch.full((1, 1), sos_token_id, dtype=torch.long, device=device)
    # BIG_OUTPUT_LOGIT_LIST = torch.empty((1, max_len, vae_model.decoder.output_dim), device=device)

    # z = z.unsqueeze(1)

    # embedded = vae_model.decoder.embedding(generated_sequence)
    # embedded = vae_model.decoder.pos_encoder(embedded)
    # tgt_mask = torch.triu(torch.ones(max_len, max_len, device=device) * float('-inf'), diagonal=1)

    # current_mask = tgt_mask[:0 + 1, :0 + 1]
    # decoded = vae_model.decoder.transformer_decoder(tgt=embedded, memory=z, tgt_mask=current_mask)

    # output_logits = vae_model.decoder.fc_out(decoded[:, -1, :])
    # BIG_OUTPUT_LOGIT_LIST[:, 0, :] = output_logits
    # temperature = 0.8
    # scaled_output_logits = output_logits / temperature
    # best_tokens = torch.topk(torch.nn.functional.softmax(scaled_output_logits, dim=-1), 15, dim=-1).indices
    # print(tokenizer.decode(best_tokens.squeeze().tolist(), skip_special_tokens=False))
        

    

    
    





    
