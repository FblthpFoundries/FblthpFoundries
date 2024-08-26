import json
import random
from gpt import gen
from local_extractor import extract_keywords
import argparse
import re
import torch
from transformers import set_seed
import requests
from openai import OpenAI
import openai
from openai import BadRequestError
import time
import uuid
from enum import Enum
from tqdm import tqdm
import logging

class SupportedModels(Enum):
    SD3 = "SD3"
    DALL_E = "DALL-E"

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(level=logging.WARNING, handlers=[TqdmLoggingHandler()])
logger = logging.getLogger(__name__)

def fetch_card_types():
    try:
        response = requests.get("https://api.scryfall.com/catalog/card-types")
        response.raise_for_status()
        return response.json()['data']
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch card types: {e}")
        raise

def fetch_creature_types():
    try:
        response = requests.get("https://api.scryfall.com/catalog/creature-types")
        response.raise_for_status()
        return response.json()['data']
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch creature types: {e}")
        raise

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

def extract_keywords_gpt(card_text):
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
        image_prompt = response.choices[0].message.content.strip()

        return True, image_prompt
    except Exception as e:
        logger.error(f"Failed to extract keywords using GPT: {e}")
        return False, None

def generate_text_local(model_path, seed=None, text_start="<tl>", max_length=400):

    if seed:
        set_seed(seed)

    # Generate card
    try:
        output = gen(
            text_start=text_start,
            max_length=max_length,
            model_path=model_path
        )
        parsed = parse_card_data(output.split("<eos>")[0])[0]
        return True, parsed
    except Exception as e:
        logger.error(f"Failed to generate text locally: {e}")
        return False, None

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
- Generate only one card per API call.
- Do not include inline formatting, specifically the ```json before and ``` after.

"""
    try:
        response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in generating Magic: The Gathering cards."},
            {"role": "user", "content": chatgpt_prompt}
        ],
        max_tokens=300
        )
        card_data = response.choices[0].message.content.strip()
        return True, card_data
    except Exception as e:
        logger.error(f"Failed to generate text using GPT: {e}")
        return False, None

def generate_image_local(prompt, model="SD3"):
    if model == "SD3":
        from diffusers import StableDiffusion3Pipeline
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3-medium-diffusers",
                    torch_dtype=torch.float16
                    )
            pipe.enable_model_cpu_offload()

            img = pipe(
                    prompt=prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=30,
                    height=800,
                    width=1024
                ).images[0]

            return img
        except Exception as e:
            logger.error(f"Failed to generate image locally: {e}")
            raise
    else:
        raise Exception(f"Model {model} not supported")

def generate_image_dalle(prompt):
    from constants import API_KEY
    openai.api_key = API_KEY

    client = OpenAI(api_key=API_KEY)
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        image_data = requests.get(image_url).content
        return True, image_data
    except BadRequestError as e:
        logger.error(f"Image rejected by safety system: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Failed to generate image using DALL-E: {e}")
        return False, None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate Magic the Gathering cards and corresponding artwork.')
    parser.add_argument('--text_start', type=str, default='<tl>', help='The text to start the card generation with. Default is "<tl>".')
    parser.add_argument('--extractor', type=str, default="local", choices=["local", "gpt"], help='The keyword extractor to use. Choose between "local" or "gpt".')
    parser.add_argument('--max_length', type=int, default=400, help='The maximum length of the generated text. Default is 400.')
    parser.add_argument('--gen_art', action='store_true', help='If set, generate art for the card.')
    parser.add_argument('--generator', type=str, default="local", choices=["local", "gpt"], help='The text generator to use. Choose between "local" or "gpt". Default is "local".')
    parser.add_argument('--model_path', type=str, default='./magic_mike/checkpoint-8500', help='The path to the model checkpoint. Default is "./magic_mike/checkpoint-8500".')
    parser.add_argument('--seed', type=int, default=None, help='The random seed for GPT-2 if used.')
    parser.add_argument('--art_model', type=str, default=SupportedModels.SD3.value, choices=[model.value for model in SupportedModels], help='The art model to prompt. Supported models are "SD3" and "DALL-E". Default is "SD3".')
    parser.add_argument('--iterations', type=int, default=1, help='The number of iterations to run. Default is 1.')

    args = parser.parse_args()
    assert args.art_model in [model.value for model in SupportedModels], f"Art model {args.art_model} is not supported. Supported models are {[model.value for model in SupportedModels]}"

    # Outer tqdm bar for tracking overall progress
    with tqdm(total=args.iterations, desc="Overall Progress", ncols=100) as overall_bar:
        # Loop over each iteration (each card generation)
        for i in range(args.iterations):
            with tqdm(total=4, desc=f"Generating Card {i+1}", ncols=100, leave=False) as card_bar:
                logger.info(f"\nStarting card {i+1}/{args.iterations}")
                
                # Start timing for card generation
                t_start_card = time.time()

                # Step 1: Generate Card
                parsed = None
                card_bar.set_description(f"Generating Card {i+1}: Text Generation")
                if args.generator == "local":
                    success, parsed = generate_text_local(args.model_path, seed=args.seed, text_start=args.text_start, max_length=args.max_length)
                elif args.generator == "gpt":
                    success, returned = generate_text_gpt()
                    if success:
                        try:
                            parsed = json.loads(returned)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                            logger.error(f"Response: {returned}")
                            continue
                    else:
                        continue
                else:
                    logger.error(f"Generator {args.generator} not supported")
                    continue
                
                card_bar.update(1)

                t_end_card = time.time()
                logger.info(f"Card generation time: {t_end_card - t_start_card:.3f}s")

                card_dict = parsed
                name = card_dict['name']

                logger.info(f"\nGenerated card {name}:")
                logger.info(json.dumps(card_dict, indent=4))

                # Step 2: Extract Artwork Prompt
                t_start_prompt = time.time()
                card_bar.set_description(f"Generating Card {i+1}: Extracting Prompt")
                
                if args.extractor == "local":
                    success, prompt = extract_keywords(parsed, model=args.art_model)
                elif args.extractor == "gpt":
                    success, prompt = extract_keywords_gpt(parsed)
                else:
                    logger.error(f"Extractor {args.extractor} not supported")
                    continue
                
                card_bar.update(1)

                t_end_prompt = time.time()
                logger.info(f"Prompt extraction time: {t_end_prompt - t_start_prompt:.3f}s")
                
                logger.info(f"\nExtracted art prompt for {name}: \n\n{prompt}")

                # Step 3: Generate Artwork (if enabled)
                if not args.gen_art:
                    logger.info("\nEnding early since gen_art was disabled.")
                    overall_bar.update(1)
                    continue

                t_start_art = time.time()
                card_bar.set_description(f"Generating Card {i+1}: Generating Artwork")
                
                img = None
                if args.art_model != SupportedModels.DALL_E.value:
                    try:
                        img = generate_image_local(prompt, model=args.art_model)
                    except Exception as e:
                        logger.error(f"Failed to generate image locally: {e}")
                        continue
                else:
                    success, img = generate_image_dalle(prompt)
                    if not success:
                        continue

                card_bar.update(1)

                t_end_art = time.time()
                logger.info(f"Artwork generation time: {t_end_art - t_start_art:.3f}s")

                # Step 4: Save the image and prompt using UUIDs
                card_bar.set_description(f"Generating Card {i+1}: Saving Files")
                file_id = uuid.uuid4()
                image_path = f'art/out/{name}_{file_id}.png'
                prompt_path = f'art/out/{name}_{file_id}.txt'
                try:
                    with open(image_path, 'wb') as file:
                        file.write(img)
                    with open(prompt_path, 'w') as file:
                        if card_dict:
                            card_dict["prompt"] = prompt
                            file.write(json.dumps(card_dict, indent=4))
                    logger.info(f"Saved image to {image_path}")
                    logger.info(f"Saved prompt to {prompt_path}")
                except Exception as e:
                    logger.error(f"Failed to save image or prompt: {e}")

                card_bar.update(1)

                t_final = time.time()
                logger.info(f"Total time for iteration {i+1}: {t_final - t_start_card:.3f}s")

                # Update the overall progress bar
                overall_bar.update(1)

        logger.info("All iterations completed.")
