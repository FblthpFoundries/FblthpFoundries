from gpt import gen
import argparse
import re
from diffusers import StableDiffusion3Pipeline
import torch
from transformers import set_seed
import requests
from constants import API_KEY
from openai import OpenAI
import time

SUPPORTED_MODELS = ["SD3", "DALL-E"]

def fetch_card_types():
    response = requests.get("https://api.scryfall.com/catalog/card-types")
    if response.status_code == 200:
        return response.json()['data']
    else:
        raise Exception("Failed to fetch card types from Scryfall")

# Function to fetch creature types from Scryfall API
def fetch_creature_types():
    response = requests.get("https://api.scryfall.com/catalog/creature-types")
    if response.status_code == 200:
        return response.json()['data']
    else:
        raise Exception("Failed to fetch creature types from Scryfall")
    
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

def extract_keywords(card_info, card_types_list, creature_types_list, model="SD3"):
    card_types = []
    subtypes = []
    description_parts = []
    intermediate = card_info['type_line'].split("—")
    if len(intermediate) == 1:
        card_types = intermediate[0].strip().split(" ")
        subtypes = []
    else:
        card_types = intermediate[0].strip().split(" ")
        subtypes = intermediate[1].strip().split(" ")

    # Identify one-word abilities from the oracle text
    abilities = {
        "flying": "soaring through the sky",
        "first strike": "ready to strike first",
        "double strike": "attacking with double force",
        "lifelink": "radiating a life-sustaining aura",
        "trample": "crushing everything underfoot",
        "deathtouch": "deadly to the touch",
        "haste": "moving with incredible speed yet sharp in focus",
        "hexproof": "surrounded by a protective aura",
        "indestructible": "impervious to damage",
        "vigilance": "ever-watchful and alert",
        "menace": "with a fearsome presence",
        "reach": "extending its reach far and wide",
        "flash": "appearing in a flash of light",
    }
    included_abilities = [description for ability, description in abilities.items() if ability in card_info['oracle_text'].lower()]

    # Identify colors from mana cost
    colors = {
        '<W>': "white",
        '<U>': "blue",
        '<B>': "black",
        '<R>': "red",
        '<G>': "green",
        '<C>': "colorless"
    }
    color_identity = []
    for symbol, color_name in colors.items():
        if symbol in card_info['mana_cost']:
            color_identity.append(color_name)

    

    # Describe size based on power/toughness
    if card_info['power'] and card_info['toughness']:
        power = int(card_info['power'])
        toughness = int(card_info['toughness'])
        avg = (power + toughness)//2

        if avg >= 8:
            size_description = "of immense size, towering over the landscape"
        elif avg >= 4:
            size_description = "of considerable size, imposing and strong"
        elif avg >= 2:
            size_description = "of medium stature"
        else:
            size_description = "smaller in stature, but agile and fierce"

        description_parts.append(size_description)

    if color_identity:
        color_identity_str = " and ".join(color_identity)
        description_parts.append(f"The image has a {color_identity_str} color theme")

    # Construct the main description
    if card_types:

        card_type_str = " ".join([k for k in card_types]).lower().strip()
        specific_type_str = " ".join([k for k in subtypes]).lower().strip()

        renames = {
            'creature': '',
            'sorcery': 'spell',
            'instant': 'cantrip spell',
            'planeswalker': 'powerful character',
            'legendary': ''
        }
        for k in renames:
            card_type_str = card_type_str.replace(k, renames[k])
        card_type_str = card_type_str.strip()

        description = (f"A {card_type_str}" if not specific_type_str else f"A {card_type_str} {specific_type_str}, {card_info['name']}")

        if included_abilities:
            ability_str = ", ".join(included_abilities)
            description += f", {ability_str}"

        description_parts.insert(0, description)

    
    # Combine all description parts into a final prompt
    prompt = ". ".join(description_parts)
    if model == "DALL-E":
        prompt += ". Rendered in high fantasy digital art style with dynamic composition and a mix of colors. Immersive landscape and scenery."
        prompt += " Do not include any text or labels in the image."
    else:
        prompt += ". Rendered in high fantasy digital art style with intricate details, dynamic composition, and a mix of dark and vibrant colors. Immersive landscape."

    # Include flavor text if available
    # if card_info['flavor_text']:
    #     prompt += f'{card_info["flavor_text"]}'


    return prompt

def extract_keywords_old(card):
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
        prompt += f" The background features a landscape. There is a themed sky as well. Use the color{"s" if len(color_words) > 1 else ""} {', '.join(color_words)}."
    else:
        prompt += f" The background features a landscape. There is a themed sky as well. Use grey colors in the art."
    

    
    if parsed['flavor_text']:
        prompt += " " + parsed['flavor_text']
    return prompt
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate Magic the Gathering cards')
    parser.add_argument('--text_start', type=str, default='<tl>', help='The text to start the generation with')
    parser.add_argument('--prompt', type=str, default=None, help='The text to start the generation with')
    parser.add_argument('--max_length', type=int, default=400, help='The maximum length of the generated text')
    parser.add_argument('--gen_art', action='store_true', help='Generate art for the card')
    parser.add_argument('--model_path', type=str, default='./magic_mike/checkpoint-8500', help='The path to the model checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--art_model', type=str, default='SD3', help='The art model to prompt')
    args = parser.parse_args()
    assert args.art_model in SUPPORTED_MODELS, f"Art model {args.art_model} is not supported. Supported models are {", ".join(SUPPORTED_MODELS)}"
    if not args.prompt:
        if args.seed:
            set_seed(args.seed)
        output = gen(
            text_start=args.text_start,
            max_length=args.max_length,
            model_path=args.model_path
        )

        parsed = parse_card_data(output.split("<eos>")[0])[0]

        print(parsed)

        [print(f"{key}: {value}") for key, value in parsed.items()]


        

        prompt = extract_keywords(parsed, fetch_card_types(), fetch_creature_types(), model=args.art_model)
        name = parsed['name']
        print(f"Extracted art prompt: {prompt}")
    else:
        prompt = args.prompt
        name = prompt[:20]
        print(f"Overriding card generation and using custom prompt: {prompt}")
        

    

    if args.gen_art:
        if args.art_model == "SD3":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                #text_encoder_3=None, #This thing absolutely annihilates my GPU, turn it off
                #tokenizer_3=None,
                torch_dtype=torch.float16
                )
            pipe.enable_model_cpu_offload()

            #pipe = pipe.to("cuda")



            img = pipe(
                    prompt=prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=30,
                    height=800,
                    width=1024
                ).images[0]
            
            img.save(f"art/SD3/{name}.png")
        if args.art_model == "DALL-E":
            print("Sending request to OpenAI API... (commonly ~15 second turnaround)")
            t1 = time.time()
            client = OpenAI(
                api_key=API_KEY
            )
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            


            # Download the image and save it
            image_data = requests.get(image_url).content
            with open(f'art/DALL-E/{name}.png', 'wb') as file:
                file.write(image_data)
            t2 = time.time()
            print(f"Image saved successfully! ({t2-t1:.3f}s)")
