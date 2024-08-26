import json
import random
from gpt import gen
import argparse
import re
import torch
from transformers import set_seed
import requests
from openai import OpenAI
import openai
from openai import BadRequestError
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

def extract_keywords(card_info, model="SD3"):

    card_types = []
    subtypes = []
    description_parts = []
    object_description = ""

    intermediate = card_info['type_line'].split("—")
    if len(intermediate) == 1:
        card_types = intermediate[0].strip().split(" ")
        subtypes = []
    else:
        card_types = intermediate[0].strip().split(" ")
        subtypes = intermediate[1].strip().split(" ")


    priority_list = ["Creature", "Artifact", "Enchantment", "Instant", "Sorcery", "Planeswalker", "Land"]
    megatype = None

    for p in priority_list:
        if p in card_types:
            megatype = p
            card_types.remove(p)
            break
    if not megatype:
        return False, None
    # if megatype != "Enchantment": #DEBUG
    #     return False, None

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
    if not color_identity:
        color_identity.append("gray")
    color_identity_str = " and ".join(color_identity)

    match megatype:
        case "Creature":
            
            if not card_info['power'] or not card_info['toughness'] or not subtypes:
                return False, None
            pronoun = "It"
            if re.search(r'\b(they|their)\b', card_info['flavor_text'], re.IGNORECASE) is not None:
                pronoun = "They" # Pandering to the WOKE MOB of LIBERALS
            if re.search(r'\b(she|her)\b', card_info['flavor_text'], re.IGNORECASE) is not None:
                pronoun = "She"
            if re.search(r'\b(he|him)\b', card_info['flavor_text'], re.IGNORECASE) is not None:
                pronoun = "He"
            
            
            # object_description = f"The foreground features a {" ".join(subtypes).lower()}, {card_info['name'].lower()}."
            object_description = f"A {" ".join(subtypes).lower()}, {card_info['name'].lower()}."
            if "Legendary" in card_types:
                object_description += " Very powerful."
            if "Artifact" in card_types:
                object_description += " Metallic in nature."
            if "Enchantment" in card_types:
                object_description += " Divine, mythical, spirit-like in nature."
            if "Land" in card_types:
                object_description += " Embodies the raw power of the land."

            
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
            if included_abilities:
                object_description += f" {pronoun} is " + ", ".join(included_abilities) + "."

            # Describe size based on power/toughness
            power = int(card_info['power'])
            toughness = int(card_info['toughness'])
            avg = (power + toughness)//2

            if avg >= 8:
                size_description = f"{pronoun} is of immense size, towering over the landscape."
            elif avg >= 4:
                size_description = f"{pronoun} is of considerable size, imposing and strong."
            elif avg >= 2:
                size_description = f"{pronoun} is of medium stature."
            else:
                size_description = f"{pronoun} is smaller in stature, but agile and fierce."

            object_description += " " + size_description
            description_parts.append(f"{pronoun} has {color_identity_str} colors.")
            description_parts.append("The background features immersive scenery.")
            description_parts.append("Rendered in high fantasy digital art style.")
            
        case "Artifact":
            
            
            if "Equipment" in subtypes:
                object_description += "" #moved into settings for more specificity
                settings = [
                    f"A warrior wielding/wearing an equipment item (weapon, tool, or armor) {card_info['name'].lower()}, featured in the foreground facing forward. The background features a realistic battlefield scene with many warriors and/or creatures across a broad landscape with various terrain features.",
                    f"An equipment item (weapon, tool, or armor) {card_info['name'].lower()} in a workshop or forge with windows. There is an artificer, metalworker, or blacksmith in the workshop. There are hints of magic and various curiosities in the workshop.",
                    f"A powerful warrior, wielding/wearing an equipment item (weapon, tool, or armor) {card_info['name'].lower()}, showcasing the power and design of the item. The item is appropriately sized for the warrior. The warrior is battling a fearsome beast. The environment is reminiscent of nature, with surrounding foliage, trees, and wildlife.",
                    f"A person in a mystical ritual or ceremony wielding/wearing an equipment item (weapon, tool, or armor) {card_info['name'].lower()}. The item is appropriately sized for the person. Small ripples, sparkles, mists, or waves of magic surround the item. The environment is a mystical temple or sacred place, that could feature water or elements of life.",

                ]
                description_parts.append(random.choice(settings))
            elif "Vehicle" in subtypes:
                settings = [
                    f"An artifact vehicle, {card_info['name'].lower()}. The vehicle is in motion, with a driver or pilot. The background features a fantastical landscape with elements of magic and fantasy.",
                    f"An artifact vehicle, {card_info['name'].lower()}. The vehicle is stationary, with a driver or pilot. The vehicle is in a workshop.",
                ]
                object_description = f""
                description_parts.append(random.choice(settings))
                description_parts.append("Reminiscent of steampunk.")
                description_parts.append(f"The vehicle has metallic color and theme, among others.")
            else:
                object_description = f"An artifact, {card_info['name'].lower()}."
                settings = [
                    f"The background features a mystical, ancient workshop, surrounded by the remnants of forgotten civilizations.",
                    f"The background features the aftermath of a scorched battlefield, glowing faintly amidst the debris and fallen warriors. Smoke rises from the earth.",
                    f"The artifact rests upon a crumbling pedestal in the heart of ancient ruins. Faint glyphs on the wall glow with a forgotten magic.",
                    f"The artifact is held by a powerful being, surrounded by a mystical aura. The background features a grand, otherworldly landscape.",
                    f"The artifact is on top of a table in a dimly lit, cluttered laboratory. Alchemical apparatus and scrolls surround it, casting long shadows. A hooded figure studies it intently from the shadows.",
                    f"The artifact sits in a glassy chamber in the depths of an ancient underwater temple, untouched by time. Schools of fish dart past, and bioluminescent plants illuminate the chamber with an ethereal glow.",
                    f"The artifact is embedded in the rocky floor of a volcanic crater, glowing intensely amidst pools of molten lava. The air is thick with ash and heat, and the sky above is a deep red, as if the artifact is feeding off the volcanic energy.",
                    f"The artifact rests on a stone altar deep within a haunted forest with twisted twees and thick fog obscuring the surroundings. Ghostly figures and whispers fill the air.",
                    f"In the center of a massive, coliseum-like arena, the artifact stands as the prize of a grand tournament. Thousands of spectators watch from the stands, and magical wards crackle around the artifact, protecting it from would-be thieves."
                    f"The artifact is half-buried in the ice of a frozen tundra, surrounded by towering glaciers and icy winds. It emits a faint, warm glow that contrasts with the bleak, cold environment, as if defying the harsh elements.",
                    f"In a lush, enchanted garden, the artifact is nestled among vibrant flowers and vines that seem to bloom unnaturally. Butterflies made of pure light flutter around it, and the air is filled with a soft, melodic hum as the artifact pulses gently.",
                    f"The artifact sits on an anvil in a dwarven forge, where molten metal flows like rivers of fire. The walls are adorned with ancient weapons and tools, and the forge's heat intensifies as the artifact's energy interacts with the surrounding flames.",
                    f"The artifact is partially submerged in the thick, murky waters of a twilight swamp. Strange creatures lurk in the shadows, and glowing eyes peer from the dark undergrowth, drawn to the artifact's eerie, phosphorescent glow.",
                    f"The artifact is held aloft by a group of robed figures in a moonlit clearing, surrounded by ancient standing stones. The air shimmers with magic, and the artifact's glow seems to pulse in time with the phases of the moon.",
                    f"The artifact is placed on a pedestal in the center of an arcane library, surrounded by towering shelves of ancient tomes and scrolls. Magic-infused orbs float around the artifact, casting a soft light, while spectral scholars peruse the endless knowledge.",
                    f"The artifact is mounted on the wall of a grand hall, surrounded by the banners and trophies of legendary heroes. The hall is filled with the echoes of past battles, and the artifact's glow illuminates the faces of the fallen warriors depicted in the tapestries.",
                    f"The artifact rests in a heavily guarded vault, surrounded by layers of magical wards and traps. The walls are made of enchanted stone, and the only light comes from the artifact itself, which pulses ominously, as if waiting for its power to be unleashed.",
                    f"The artifact is held by a powerful sorcerer atop a towering spire, overlooking a sprawling city below. The sky is filled with swirling storm clouds, and bolts of lightning arc around the artifact, crackling with raw magical energy.",
                    f"The artifact is enshrined in a sacred temple, surrounded by offerings and prayers from devoted worshippers. The air is thick with incense, and the temple's walls are adorned with intricate carvings that depict the artifact's legendary history.",
                    f"The artifact is securely fastened to the deck of a skyship, sailing through stormy skies. The scene takes place on the deck of the skyship. Lightning crackles around the ship's hull, and the artifact seems to absorb the energy, glowing brighter with each bolt that strikes the vessel.",
                    f"The artifact is hidden deep within a labyrinthine dungeon, guarded by traps, monsters, and ancient curses. The walls are lined with the bones of fallen adventurers, and the air is thick with the stench of decay and magic.",
                    f"The artifact hovers at the center of a temporal rift, where time and space twist and blur. Fragments of past and future events flash around it, and the artifact glows with a strange light, holding the unstable rift together with its arcane power.",
                    f"The artifact is enshrined in a tranquil mountain monastery, high above the clouds. Monks chant softly in the background, and the artifact is bathed in the golden light of the setting sun, radiating peace and serenity.",
                    f"The artifact is held by a powerful dragon in its hoard, surrounded by mountains of gold and jewels. The dragon's eyes gleam with avarice, and the artifact's glow seems to intensify as the dragon's greed grows.",
                ]
                description_parts.append(random.choice(settings))
                description_parts.append(f"The artifact has metallic colors, among others.")

            if "Legendary" in card_types:
                object_description += "The item is very powerful and ornate."
            description_parts.append("Rendered in high fantasy digital art style.")
            
            
        case "Enchantment":
            object_description = f"The foreground features an enchantment, reminiscent of {card_info['name'].lower()}."
            if "Aura" in subtypes:
                if "enchant " not in card_info['oracle_text'].lower():
                    return False, None
                targets = {
                    "creature": "person",
                    "land": "land",
                    "artifact": "artifact",
                    "planeswalker": "powerful being",
                    "player": "powerful being"
                }
                target = card_info['oracle_text'].lower().split('enchant ')[1].strip()
                target = target.split(" ")[0]
                object_description = f"A {targets[target]} reminiscent of fantasy surrounded or affected by an aura of enchantment. The enchantment is strongly reminiscent of {card_info['name'].lower()}."
            if "Legendary" in card_types:
                object_description += " Very powerful and mystical."
            
            description_parts.append(f"The enchantment has some {color_identity_str} color themes.")
            if "green" in color_identity_str:
                description_parts.append("Forestry, nature, woods, life, growth, and the cycle of life can be minor themes in the art.")
            if "blue" in color_identity_str:
                description_parts.append("Oceans, water, skies, dreams, and illusions can be minor themes in the art.")
            if "red" in color_identity_str:
                description_parts.append("Fire, passion, chaos, and destruction can be minor themes in the art.")
            if "black" in color_identity_str:
                description_parts.append("Death, decay, darkness, and the macabre can be minor themes in the art.")
            if "white" in color_identity_str:
                description_parts.append("Order, purity, light, and protection can be minor themes in the art.")
            if "gray" in color_identity_str:
                description_parts.append("Devoid, neutrality, and the void can be minor themes in the art.")
            description_parts.append("The background features a mystical landscape hinting at the presence of powerful, ongoing magic.")
            description_parts.append("Rendered in high fantasy digital art illustration style with dynamic composition and a mix of colors. The image is simplistic.")
            
        case "Instant":
            object_description = f"A caster casting a burst spell, {card_info['name'].lower()}."
            description_parts.append(f"The spell has a {color_identity_str} color theme.")
            description_parts.append("The background features a mystical environment reacting to the sudden burst of energy from the spell.")
            description_parts.append("Rendered in high fantasy digital art style with dynamic composition and a mix of colors.")
            
        case "Sorcery":
            object_description = f"A caster casting a powerful spell, {card_info['name'].lower()}."
            description_parts.append(f"The spell has a {color_identity_str} color theme.")
            description_parts.append("The background features a grand environment, with the ground and air altered by the massive spell being cast.")
            description_parts.append("Rendered in high fantasy digital art style with dynamic composition and a mix of colors.")
            
        case "Planeswalker":
            object_description = f"An imposing figure, {card_info['name'].lower()}, crackling with magical energy, eyes glowing."
            description_parts.append(f"It has a {color_identity_str} colored magic and color theme.")
            description_parts.append("The background features an epic, otherworldly landscape.")
            description_parts.append("Rendered in high fantasy digital art style with dynamic composition and a mix of colors.")
            
    
    

    prompt = object_description  + (" " if description_parts else "") +  " ".join(description_parts)

    
    # Combine all description parts into a final prompt
    if model == "DALL-E":
        pass
        #prompt += " Do not include any text or labels in the image."
    else:
        pass

    # Include flavor text if available
    # if card_info['flavor_text']:
    #     prompt += f'{card_info["flavor_text"]}'


    return True, prompt

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

def extract_keywords_ai(card_text):
    chatgpt_prompt = f"""
    Given the following Magic: The Gathering card text, generate an image prompt for DALL-E to create an image that captures the essence of the card.:

    Type Line: {card_text['type_line']}
    Name: {card_text['name']}
    Mana Cost: {card_text['mana_cost']}
    Oracle Text: {card_text['oracle_text']}
    Power: {card_text['power']}
    Toughness: {card_text['toughness']}
    Loyalty: {card_text['loyalty']}
    Flavor Text: {card_text['flavor_text']}

    Please provide a detailed description for an image that captures the essence of this card. Avoid text in the image. 
    If there are named characters from MTG in the name, oracle text, or flavor text, do your best to incorporate them into the art.
    Simply explain what to generate.
    Avoid using the words "token" and "card" in the prompt. Avoid urban environments, unless they are somewhat fantasy in nature.
    Return ONLY the prompt as a response. The prompt should be summarized in 150 tokens.
    """
    from constants import API_KEY
    openai.api_key = API_KEY
    # Send the request to ChatGPT
    try:
        response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generating descriptive art prompts."},
            {"role": "user", "content": chatgpt_prompt}
        ],
        max_tokens=175
        )
        #print(response)
        image_prompt = response.choices[0].message.content.strip()

        return True, image_prompt
    except:
        return False, None

def generate_text_local(model_path, seed=None, text_start="<tl>", max_length=400):

    if seed:
        set_seed(seed)

    #generate card
    output = gen(
        text_start=text_start,
        max_length=max_length,
        model_path=model_path
    )

    parsed = parse_card_data(output.split("<eos>")[0])[0]

    return True, parsed

def generate_text_gpt(model="gpt-4o-mini"):
    from constants import API_KEY
    openai.api_key = API_KEY
    chatgpt_prompt = f"""
Create a unique card within the world of Magic: The Gathering, ensuring it follows the game's official rules. This card should have a diverse set of characteristics, including a randomly chosen color identity, mana cost, and type with appropriate subtypes.

- Choose a random color identity and design a mana cost that reflects it, with colorless cost appearing before colored cost.
- Select a random type and subtype, making sure it fits within a themed and cohesive narrative.
- Ensure that this card could seamlessly integrate into a well-rounded strategy, with a power level that complements the broader game.
- Return ONLY a JSON dictionary with 'type_line', 'name', 'mana_cost', 'oracle_text', 'flavor_text', 'power', 'toughness', and 'loyalty' fields, with string values.
- If a field is not applicable, set it to an empty string.
- Format mana symbols with angle brackets, e.g., <G>.
- Focus on thematic elements and do not explicitly mention "cards" or "tokens".
- Generate only one entity per API call.
"""
    from constants import API_KEY
    openai.api_key = API_KEY
    # Send the request to ChatGPT
    try:
        response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generating Magic: The Gathering cards."},
            {"role": "user", "content": chatgpt_prompt}
        ],
        max_tokens=300
        )
        image_prompt = response.choices[0].message.content.strip()

        return True, image_prompt
    except:
        return False, None

def generate_image_local(prompt, model="SD3"):
    if model == "SD3":
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                #text_encoder_3=None, #This thing absolutely annihilates my GPU, turn it off
                #tokenizer_3=None,
                torch_dtype=torch.float16
                ) #TODO: cache model between iterations to save performance
        
        pipe.enable_model_cpu_offload()

        #pipe = pipe.to("cuda")

        img = pipe(
                prompt=prompt,
                num_images_per_prompt=1,
                num_inference_steps=30,
                height=800,
                width=1024
            ).images[0]
        
        return img
    else:
        raise Exception(f"Model {model} not supported")

def generate_image_dalle(prompt):
    try:
        from constants import API_KEY
    except:
        raise Exception("constants.py not found. Please rename constants_example.py to constants.py and add your OpenAI API key to use DALL-E for image generation.")
    
    assert API_KEY != "your-api-key-here", "Please add your OpenAI API key to constants.py"
    print("Sending request to OpenAI API... (commonly ~15 second turnaround)")
    client = OpenAI(
        api_key=API_KEY
    )
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )

        image_url = response.data[0].url
    except BadRequestError as e:
        print(f"Image rejected by safety system. Skipping...")
        return False, None
    
    # Download the image and save it
    image_data = requests.get(image_url).content
    return True, image_data

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate Magic the Gathering cards')
    parser.add_argument('--text_start', type=str, default='<tl>', help='The text to start the generation with')
    parser.add_argument('--extractor', type=str, default="local", help="The keyword extractor to use")
    parser.add_argument('--max_length', type=int, default=400, help='The maximum length of the generated text')
    parser.add_argument('--gen_art', action='store_true', help='Generate art for the card')
    parser.add_argument('--generator', type=str, default="local", help='The text generator to use')
    parser.add_argument('--model_path', type=str, default='./magic_mike/checkpoint-8500', help='The path to the model checkpoint')
    parser.add_argument('--seed', type=int, default=None, help='The random seed for gpt2 if used')
    parser.add_argument('--art_model', type=str, default='SD3', help='The art model to prompt')
    parser.add_argument('--iterations', type=int, default=1, help='The number of iterations to run')
    args = parser.parse_args()
    assert args.art_model in SUPPORTED_MODELS, f"Art model {args.art_model} is not supported. Supported models are {", ".join(SUPPORTED_MODELS)}"

    for i in range(args.iterations):

        print(f"\nStarting card {i+1}/{args.iterations}")
        
        # Start timing for card generation
        t_start_card = time.time()

        # Generate Card
        parsed = None
        if args.generator == "local":
            success, parsed = generate_text_local(args.model_path, seed=args.seed, text_start=args.text_start, max_length=args.max_length)
        elif args.generator == "gpt":
            success, returned = generate_text_gpt()
            try:
                parsed = json.loads(returned)
            except json.JSONDecodeError:
                print("Failed to parse JSON response")
                print(returned)
        else:
            raise Exception(f"Generator {args.generator} not supported")
        
        t_end_card = time.time()
        print(f"Card generation time: {t_end_card - t_start_card:.3f}s")

        card_dict = parsed
        name = card_dict['name']

        print(f"\nGenerated card {name}:")
        print(json.dumps(card_dict, indent=4))

        # Start timing for prompt extraction
        t_start_prompt = time.time()
        
        # Generate Artwork prompt
        if args.extractor == "local":
            success, prompt = extract_keywords(parsed, model=args.art_model)
        elif args.extractor == "gpt":
            success, prompt = extract_keywords_ai(parsed)
        else:
            raise Exception(f"Extractor {args.extractor} not supported")
        
        t_end_prompt = time.time()
        print(f"Prompt extraction time: {t_end_prompt - t_start_prompt:.3f}s")
        
        print(f"\nExtracted art prompt for {name}: \n\n{prompt}")

        # End early if artwork generation is disabled
        if not args.gen_art:
            print("\nEnding early since gen_art was disabled.")
            continue

        # Start timing for artwork generation
        t_start_art = time.time()

        # Generate Artwork
        print("\nBeginning artwork generation!")
        
        if args.art_model != "DALL-E":
            img = generate_image_local(prompt, model=args.art_model)
        else:
            success, img = generate_image_dalle(prompt)

        t_end_art = time.time()
        print(f"Artwork generation time: {t_end_art - t_start_art:.3f}s")

        # Save the image and prompt
        with open(f'art/out/{name}.png', 'wb') as file:
            file.write(img)
        with open(f'art/out/{name}.txt', 'w') as file:
            if card_dict:
                card_dict["prompt"] = prompt
                file.write(json.dumps(card_dict, indent=4))

        t_final = time.time()
        print(f"Total time for iteration {i+1}: {t_final - t_start_card:.3f}s")
