from gpt import gen
import argparse
import re
from diffusers import StableDiffusion3Pipeline
import torch
from transformers import set_seed

def parse_card_data(input_text):
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
        card['type_line'] = re.search(card_pattern, card_match).group(1).strip()
        
        name = re.search(name_pattern, card_match)
        card['name'] = name.group(1).strip() if name else None
        
        mc = re.search(mc_pattern, card_match)
        card['mana_cost'] = mc.group(1).strip() if mc else None
        
        ot = re.search(ot_pattern, card_match)
        card['oracle_text'] = re.sub(r'<nl>', '\n', ot.group(1).strip()) if ot else None
        card['oracle_text'] = card['oracle_text'].replace('<br>', '\n')
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
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate Magic the Gathering cards')
    parser.add_argument('--text_start', type=str, default='<tl>', help='The text to start the generation with')
    parser.add_argument('--max_length', type=int, default=400, help='The maximum length of the generated text')
    parser.add_argument('--gen_art', action='store_true', help='Generate art for the card')
    parser.add_argument('--model_path', type=str, default='./magic_mike/checkpoint-8500', help='The path to the model checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    args = parser.parse_args()
    if args.seed:
        set_seed(args.seed)
    output = gen(
        text_start=args.text_start,
        max_length=args.max_length,
        model_path=args.model_path
    )
    #output = "<tl> Creature — Vampire Rogue <\\tl> <name> Vengeful Rogue <\\name> <mc> <3> <B> <\\mc> <ot> <T>: Draw a card, then discard a card. <\\ot> <power> 4 <\\power> <toughness> 4 <\\toughness> <loyalty> <\\loyalty> <ft> The greatest evil lies behind all the shadows and shadows in my mind, and all I fear is not death, but only peace. <nl> —The Dark Path of the Eternals <\\ft><eos><tl> Creature — Bird Soldier <\\tl> <name> Gullwing Pilot <\\name> <mc> <4> <W> <\\mc> <ot> Flying <nl> Whenever ~ deals combat damage to a player, destroy target nonland permanent. <\\ot> <power> 3 <\\power> <toughness> 3 <\\toughness> <loyalty> <\\loyalty> <ft> When not flying, he rides to the skies. <nl> —Oedipus <\\ft><eos><tl> Artifact <\\tl> <name> Sword of the Eternals <\\name> <mc> <3> <\\mc> <ot> <T>: Target creature you control gets +2/+2 until end of turn. <\\ot> <power> <\\power> <toughness> <\\toughness> <loyalty> <\\loyalty> <ft> This sword is the first gift we have given to ourselves in this world. <\\ft><eos><tl> Legendary Creature — Shapeshifter <\\tl> <name> Chandra, Witch's Apprentice <\\name> <mc> <X> <U> <B> <G"
    #print(output)
    #bs = BeautifulSoup(output, 'lxml')
    #print(bs.get_text())
    parsed = parse_card_data(output.split("<eos>")[0])[0]

    [print(f"{key}: {value}") for key, value in parsed.items()]


    pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None, #This thing absolutely annihilates my GPU, turn it off
    #tokenizer_3=None,
    torch_dtype=torch.float16
    )

    pipe = pipe.to("cuda")

    colormap = {
        "White": "<W>",
        "Blue": "<U>",
        "Black": "<B>",
        "Red": "<R>",
        "Green": "<G>",
    }
    color_words = [color for color in colormap.keys() if colormap[color] in parsed['mana_cost']]
    if "—" in parsed['type_line']:
        type, subtype = parsed['type_line'].split("—")
        subtype = subtype.strip()
        type = type.strip()
    else:
        type = parsed['type_line']
        type = type.strip()
        subtype = None

    if type == "Creature":
        prompt = f"Fantasy digital art of a character called {parsed['name']}. It is a {subtype}."
    elif type == "Enchantment":
        prompt = f"Fantasy digital art of an enchantment called {parsed['name']}.{"" if not subtype else f' It is a {subtype}.'}"
    elif type == "Artifact":
        prompt = f"Fantasy digital art of an enchantment called {parsed['name']}.{"" if not subtype else f' It is a {subtype}.'}"
    elif type == "Instant" or type == "Sorcery":
        prompt = f"Fantasy digital art of a spell called {parsed['name']}.{"" if not subtype else f' It is a {subtype}.'}"
    else:
        prompt = f"Fantasy digital art of {parsed['name']}, a {type}."


    if parsed['power'] and parsed['toughness']:
        size = (int(parsed["power"]) + int(parsed["toughness"]) )// 2
        if size >= 10:
            prompt += " It is colossal."
        elif size >= 5:
            prompt += " It is large."
        elif size <= 2:
            prompt += " It is small."
        else:
            prompt += " It is medium sized."

    color = ', '.join(color_words)
    if color:
        prompt += f" The background features a landscape with the color{"s" if len(color_words) > 1 else ""} {', '.join(color_words)}. There is a sky as well. Keep the background neutral."
    else:
        prompt += f" The background features a landscape with the color grey. There is a sky as well. Keep the background neutral."
    

    
    if parsed['flavor_text']:
        prompt += " " + parsed['flavor_text']


    print()

    print(prompt)
    if args.gen_art:
        img = pipe(
                prompt=prompt,
                negative_prompt="cartoon",
                num_images_per_prompt=1,
                num_inference_steps=35,
                height=800,
                width=1024
            ).images[0]
        
        img.save(f"art/{parsed['name']}.png")
