import json
import os
import random
import threading
import queue
import uuid
import openai
import requests
import re
import torch
import argparse
import time
import logging
from enum import Enum
from tqdm import tqdm
from constants import PROXYSHOP_PATH
from openai import OpenAI, BadRequestError
from helpers.distribution import seed_card
from helpers.local_prompt_extractor import extract_keywords


class SupportedModels(Enum):
    SD3 = "SD3"
    DALL_E = "DALL-E"
    DALL_E_WIDE = "DALL-E-WIDE"


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


class FblthpFoundries:
    def __init__(self, args):
        self.args = args
        self.render_queue = queue.Queue()
        self.gpt_model = "gpt-4o-mini"
        self.original_wd = os.getcwd()
        self.outfolder = os.path.join(self.original_wd, "out")
        self.runfolder = f"run{self.get_next_run_number(self.outfolder)}"
        self.runfolder = os.path.join(self.outfolder, self.runfolder)
        self.resource_folder = os.path.join(self.runfolder, "resources")
        os.mkdir(self.runfolder)
        os.mkdir(self.resource_folder)

        logging.basicConfig(level=logging.WARNING, handlers=[TqdmLoggingHandler()])
        self.logger = logging.getLogger(__name__)

        # Start the rendering thread
        self.render_thread = threading.Thread(target=self.render_process, daemon=True, args=(self.runfolder,))
        self.render_thread.start()

    def parse_card_data(self, input_text):
        """
        Parses the input text and extracts card data.
        To be used on output coming from the GPT-2 model.
        Args:
            input_text (str): The input text containing card data.
        Returns:
            list: A list of dictionaries, where each dictionary represents a card and contains the following keys:
                - 'type_line': The type line of the card.
                - 'name': The name of the card.
                - 'mana_cost': The mana cost of the card.
                - 'oracle_text': The oracle text of the card.
                - 'power': The power of the card.
                - 'toughness': The toughness of the card.
                - 'loyalty': The loyalty of the card.
                - 'flavor_text': The flavor text of the card.
        """
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

    def extract_keywords_gpt(self, card_text, model="gpt-4o-mini"):
        '''
        Extracts keywords from Magic: The Gathering card text and generates an image prompt for DALL-E to create an image that captures the essence of the card.
        Args:
            card_text (dict): A dictionary containing the following card information:
                - 'type_line' (str): The type line of the card.
                - 'name' (str): The name of the card.
                - 'mana_cost' (str): The mana cost of the card.
                - 'oracle_text' (str): The oracle text of the card.
                - 'power' (str): The power of the card.
                - 'toughness' (str): The toughness of the card.
                - 'loyalty' (str): The loyalty of the card.
                - 'flavor_text' (str): The flavor text of the card.
                - 'rarity' (str): The rarity of the card.
            model (str, optional): The model to use for generating the image prompt. Defaults to "gpt-4o-mini".
        Returns:
            tuple: A tuple containing a boolean indicating the success of the operation and the generated image prompt.
        '''
        prompt = f"""
        Given the following Magic: The Gathering card text, generate an image prompt for DALL-E to create an image that captures the essence of the card.:

        Type Line: {card_text['type_line']}
        Name: {card_text['name']}
        Mana Cost: {card_text['mana_cost']}
        Oracle Text: {card_text['oracle_text']}
        Power: {card_text['power']}
        Toughness: {card_text['toughness']}
        Loyalty: {card_text['loyalty']}
        Flavor Text: {card_text['flavor_text']}
        Rarity: {card_text['rarity']}

        Please provide a detailed description for an image that captures the essence of this card. Avoid text in the image. 
        Simply explain what to generate. Do not mention Magic: The Gathering.
        {"Artifacts should have metallic elements." if "Artifact" in card_text['type_line'] else ""}
        Don't use the words "token", "permanent", "counter", or "card" in the prompt. Instead suggest specific creatures, humans, or objects in their place. Avoid urban environments, unless they are somewhat fantasy in nature.
        Return only JSON as output with the returned prompt stored in a "prompt" field. The prompt should be summarized in 150 tokens.
        """
        from constants import API_KEY
        openai.api_key = API_KEY
        # Send the request to ChatGPT
        success, out = self.ask_gpt(prompt, model=model)
        image_prompt = out["prompt"]
        return True, image_prompt

    def generate_text_local(self, model_path, seed=None, text_start="<tl>", max_length=400):
        """
        Generates text locally using a pre-trained model.
        Args:
            model_path (str): The path to the pre-trained model.
            seed (int, optional): The random seed for text generation. Defaults to None.
            text_start (str, optional): The starting text for generation. Defaults to "<tl>".
            max_length (int, optional): The maximum length of the generated text. Defaults to 400.
        Returns:
            tuple: A tuple containing a boolean indicating success or failure of text generation, and the parsed generated text.
        """

        from transformers import set_seed
        from gpt import gen
        if seed:
            set_seed(seed)

        # Generate card
        try:
            output = gen(
                text_start=text_start,
                max_length=max_length,
                model_path=model_path
            )
            parsed = self.parse_card_data(output.split("<eos>")[0])[0]
            parsed['rarity'] = random.choice(["Common", "Uncommon", "Rare", "Mythic Rare"])  # TODO: fix
            return True, parsed
        except Exception as e:
            self.logger.error(f"Failed to generate text locally: {e}")
            return False, None

    def ask_gpt(self, prompt, header="You are an expert in generating Magic: The Gathering cards.", model="gpt-4o-mini"):
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": header},
                {"role": "user", "content": prompt}
            ],
        )
        resp = response.choices[0].message.content.strip()
        if '```' in resp:
            resp = resp.split('```')[1]
        if resp[:4] == "json":
            resp = resp[4:]
        return True, json.loads(resp)

    def generate_card_gpt(self, model="gpt-4o-mini"):
        from constants import API_KEY
        openai.api_key = API_KEY

        card_type, colors, mana_cost, rarity = seed_card()

        color_identity = ", ".join(colors) if colors else "Colorless"
        subbie = None
        if "Artifact" in card_type:
            if "Vehicle" not in card_type:
                subbie = f"""{('- The card is a mana rock.' if random.random() < 0.4 
                else ("- The card has an ability with {T} as part of the cost." if random.random() < 0.5 
                else "")) if 'Artifact' in card_type else ''}"""
            else:
                subbie = "The card is a vehicle"
        theme_prompt = f'''
        You will be creating a unique card within the world of Magic: The Gathering. 
    Start with the following characteristics of the card:

    - Color Identity: {color_identity if color_identity else "Colorless"}
    - Card Type: {card_type}
    - Mana Cost: {mana_cost}
    - Type: {card_type}
    {subbie if subbie else ""}
    - Rarity: {rarity}

    - Generate 20 themes for the card based on these characteristics. These go in the 'themes' field.
    - Return a JSON dictionary with a 'themes' field, with a list of string values.
    '''

        try:
            success, themes = self.ask_gpt(theme_prompt, model=model)
            if not success:
                raise Exception("Failed to generate themes using GPT")
            themes = themes['themes']
            theme = random.choice(themes)

            name_prompt = f'''
        You will be creating a unique card within the world of Magic: The Gathering. 
    Start with the following characteristics of the card:

    - Color Identity: {color_identity if color_identity else "Colorless"}
    - Card Type: {card_type}
    - Mana Cost: {mana_cost}
    - Type: {card_type}
    {subbie if subbie else ""}
    - Rarity: {rarity}
    - Theme: {theme}

    - Generate 20 names for the card based on these characteristics. Ensure the name makes sense for a {card_type} card. These go in the 'names' field.
    - Return a JSON dictionary with a 'names' field, with a list of string values.
    '''
            success, names = self.ask_gpt(name_prompt, model=model)
            if not success:
                raise Exception("Failed to generate names using GPT")
            names = names['names']
            name = random.choice(names)

            strength_dict = {
                "common": "somewhat weak and have only one ability",
                "uncommon": "reasonably balanced with good synergy",
                "rare": "strong, with synergistic abilities",
                "mythic rare": "a strong, high-impact card with multiple syngergistic abilities"
            }
            chatgpt_prompt = f"""
        Create a unique card within the world of Magic: The Gathering, ensuring it follows the game's official rules. 

        Start with the following characteristics of the card:
        - Name: {name}
        - Theme: {theme}
        - Color Identity: {color_identity if color_identity else "Colorless"}
        - Card Type: {card_type}
        - Mana Cost: {mana_cost}
        - Type: {card_type}
        - Rarity: {rarity}
        
        When designing the card, follow these rules:
        
        {'- The card should have creature subtypes fitting its theme.' if 'Creature' in card_type else ''}
        {'- The card should have empty fields for power and toughness.' if ('Creature' not in card_type and 'Vehicle' not in card_type) else ''}
        {'- Planeswalkers should have a loyalty value in the "loyalty" field, as well as abilities that reflect their character. Planeswalkers also have a subtype with their first name in the type line. i.e. Planeswalker - <firstname>' if 'Planeswalker' in card_type else ''}
        {'- Lands should have no mana cost, and have at least one ability that produces mana.' if 'Land' in card_type else ''}
        {subbie if subbie else ""}
        {"- The card must have a crew ability with a cost." if "Vehicle" in card_type else ""}
        - Make it {strength_dict[rarity.lower()]}
        - Place separate abilities on new lines.
        - Cards should have flavor text.
        {"Give the card a new keyword ability." if (random.random() < 0.3) else ""}
        - Return a JSON dictionary with 'type_line', 'name', 'mana_cost', 'rarity', 'oracle_text', 'flavor_text', 'power', 'toughness', and 'loyalty' fields, with string values. Each of these values must be present in the dictionary.
        - If a field is not applicable, set it to an empty string.
        - Format mana symbols with curly braces.
        - Generate only one card per API call.
        """
            success, initial_card = self.ask_gpt(chatgpt_prompt, model=model)
            if not success:
                raise Exception("Failed to generate card using GPT")

            balancing_prompt = f"""{initial_card}\n
        This is JSON for a generated Magic: the gathering card. I would like you to read the card, and adjust its abilities to follow the rules of the game.
        - Return a JSON dictionary with the adjusted 'oracle_text' field, with a string value.
        - Also include a field 'adjustment' in the dictionary for an explanation of the changes you made and why. This JSON output is all I want.
    """
            success, new_card = self.ask_gpt(balancing_prompt, model=model)
            if not success:
                raise Exception("Failed to balance card using GPT")
            card = initial_card.copy()
            card['oracle_text'] = new_card['oracle_text']
            card['adjustment'] = new_card['adjustment']
            card['theme'] = theme
            card['themes'] = themes
            card['names'] = names
            card['initial-card'] = initial_card
            card['chatgpt-prompt'] = chatgpt_prompt

            return True, card
        except Exception as e:
            self.logger.error(f"Failed to generate text using GPT: {e}")
            return False, None

    def generate_image_local(self, prompt, model="SD3"):
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
                self.logger.error(f"Failed to generate image locally: {e}")
                raise
        else:
            raise Exception(f"Model {model} not supported")

    def generate_image_dalle(self, prompt, model="DALL-E", quality="standard"):
        from constants import API_KEY
        openai.api_key = API_KEY

        client = OpenAI(api_key=API_KEY)
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=("1024x1024" if model == "DALL-E" else "1792x1024"),
                quality=quality,
                n=1
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            return True, image_data
        except BadRequestError as e:
            self.logger.error(f"Image rejected by safety system: {e}")
            return False, None
        except Exception as e:
            self.logger.error(f"Failed to generate image using DALL-E: {e}")
            return False, None

    def generate_card_process(self, i, overall_bar, total, quality):
        with tqdm(total=total, desc=f"Generating Card {i + 1}", ncols=100, leave=False) as card_bar:
            self.logger.info(f"\nStarting card {i + 1}/{self.args.iterations}")

            # Start timing for card generation
            t_start_card = time.time()
            # Step 1: Generate Card
            parsed = None
            card_bar.set_description(f"Generating Card {i + 1}: Text Generation")
            if self.args.generator == "local":
                success, parsed = self.generate_text_local(self.args.model_path, seed=self.args.seed, text_start=self.args.text_start, max_length=self.args.max_length)
            elif self.args.generator == "gpt":
                success, returned = self.generate_card_gpt(model=self.gpt_model)
                if success:
                    try:
                        parsed = returned
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse JSON response: {e}")
                        self.logger.error(f"Response: {returned}")
                        return
                else:
                    return
            else:
                self.logger.error(f"Generator {self.args.generator} not supported")
                return

            card_bar.update(1)

            t_end_card = time.time()
            self.logger.info(f"Card generation time: {t_end_card - t_start_card:.3f}s")

            card_dict = parsed
            card_dict["gpt-model"] = self.gpt_model
            card_dict["image_generator"] = self.args.art_model
            name = card_dict['name']

            self.logger.info(f"\nGenerated card {name}:")
            self.logger.info(json.dumps(card_dict, indent=4))

            # Step 2: Extract Artwork Prompt
            t_start_prompt = time.time()
            card_bar.set_description(f"Generating Card {i + 1}: Extracting Prompt")

            if self.args.extractor == "local":
                success, prompt = extract_keywords(parsed, model=self.args.art_model)
            elif self.args.extractor == "gpt":
                success, prompt = self.extract_keywords_gpt(parsed, model=self.gpt_model)
            else:
                self.logger.error(f"Extractor {self.args.extractor} not supported")
                return

            card_bar.update(1)

            t_end_prompt = time.time()
            self.logger.info(f"Prompt extraction time: {t_end_prompt - t_start_prompt:.3f}s")

            self.logger.info(f"\nExtracted art prompt for {name}: \n\n{prompt}")

            # Step 3: Generate Artwork (if enabled)
            # Save the card data as JSON regardless of the --gen_art flag
            file_id = uuid.uuid4()
            card_dict["uuid"] = str(file_id)
            json_path = f'{name}_{file_id}.json'
            json_path = os.path.join(self.runfolder, json_path)
            try:
                with open(json_path, 'w') as file:
                    card_dict["prompt"] = prompt
                    file.write(json.dumps(card_dict, indent=4))
                self.logger.info(f"Saved card data to {json_path}")
            except Exception as e:
                self.logger.error(f"Failed to save JSON data: {e}")

            if not self.args.gen_art:
                self.logger.info("\nEnding early since gen_art was disabled.")
                overall_bar.update(1)
                return

            t_start_art = time.time()
            card_bar.set_description(f"Generating Card {i + 1}: Generating Artwork")

            img = None
            if self.args.art_model not in [SupportedModels.DALL_E.value, SupportedModels.DALL_E_WIDE.value]:
                try:
                    img = self.generate_image_local(prompt, model=self.args.art_model)
                except Exception as e:
                    self.logger.error(f"Failed to generate image locally: {e}")
                    return
            else:
                success, img = self.generate_image_dalle(prompt, model=self.args.art_model, quality=quality)
                if not success:
                    return

            card_bar.update(1)

            t_end_art = time.time()
            self.logger.info(f"Artwork generation time: {t_end_art - t_start_art:.3f}s")

            # Step 4: Save the image and prompt using UUIDs
            card_bar.set_description(f"Generating Card {i + 1}: Saving Files")
            image_path = f'{name}_{file_id}.png'
            image_path = os.path.join(self.runfolder, image_path)
            try:
                with open(image_path, 'wb') as file:
                    file.write(img)
                self.logger.info(f"Saved image to {image_path}")
            except Exception as e:
                self.logger.error(f"Failed to save image or prompt: {e}")

            # Add the card to the render queue
            self.render_queue.put((card_dict, image_path))

            card_bar.update(1)
            t_final = time.time()
            self.logger.info(f"Total time for iteration {i + 1}: {t_final - t_start_card:.3f}s")

            # Update the overall progress bar
            overall_bar.update(1)

    def render_process(self, runfolder):
        while True:
            card_dict, image_path = self.render_queue.get()
            # print(card_dict)
            if card_dict is None:  # Exit signal
                self.render_queue.task_done()
                break

            self.logger.info(f"Rendering card: {card_dict['name']}")
            from helpers.render import render_card

            render_card(card_dict, image_path)

            target = os.path.join(PROXYSHOP_PATH, "out")
            for filename in os.listdir(target):
                if filename.startswith(card_dict["name"]):
                    new_path = os.path.join(runfolder, f"{card_dict['name']}_{card_dict['uuid']}_FINAL.png")
                    os.rename(os.path.join(target, filename), new_path)
                    self.logger.info(f"Rendered card saved to {new_path}")
                    break
            self.render_queue.task_done()

    def get_next_run_number(self, directory):
        # Regular expression to match "runN" where N is an integer
        pattern = re.compile(r'^run(\d+)$')

        # Initialize a variable to keep track of the largest N found
        max_n = 0

        # Loop through the items in the directory
        for item in os.listdir(directory):
            # Check if the item is a directory and matches the pattern
            match = pattern.match(item)
            if match and os.path.isdir(os.path.join(directory, item)):
                # Extract the number N and convert it to an integer
                n = int(match.group(1))
                # Update max_n if this N is larger than the previous ones
                if n > max_n:
                    max_n = n

        # Return N+1 where N is the largest found
        return max_n + 1

    def run(self):
        # Outer tqdm bar for tracking overall progress
        with tqdm(total=self.args.iterations, desc="Overall Progress", ncols=100) as overall_bar:
            for i in range(self.args.iterations):
                total = 4  # Adjust based on your steps
                if self.args.insta_render:
                    total += 1

                # Create a thread for the card generation process
                self.generate_card_process(i, overall_bar, total, self.args.quality)

        # Signal the rendering thread to exit
        self.render_queue.put((None, None))
        self.render_queue.join()

        self.logger.info("All iterations and renderings completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Magic the Gathering cards and corresponding artwork.')
    parser.add_argument('--text_start', type=str, default='<tl>', help='The text to start the card generation with. Default is "<tl>".')
    parser.add_argument('--extractor', type=str, default="gpt", choices=["local", "gpt"], help='The keyword extractor to use. Choose between "local" or "gpt".')
    parser.add_argument('--max_length', type=int, default=400, help='The maximum length of the generated text. Default is 400.')
    parser.add_argument('--gen_art', action='store_true', help='If set, generate art for the card.')
    parser.add_argument('--generator', type=str, default="gpt", choices=["local", "gpt"], help='The text generator to use. Choose between "local" or "gpt". Default is "gpt".')
    parser.add_argument('--model_path', type=str, default='./magic_mike/checkpoint-8500', help='The path to the model checkpoint. Default is "./magic_mike/checkpoint-8500".')
    parser.add_argument('--seed', type=int, default=None, help='The random seed for GPT-2 if used.')
    parser.add_argument('--art_model', type=str, default=SupportedModels.DALL_E.value, choices=[model.value for model in SupportedModels], help='The art model to prompt. Supported models are "SD3" and "DALL-E"/"DALL-E-WIDE". Default is "DALL-E".')
    parser.add_argument('--iterations', type=int, default=1, help='The number of iterations to run. Default is 1.')
    parser.add_argument('--insta_render', action='store_true', help='If set, render the card immediately.')
    parser.add_argument('--quality', type=str, default="standard", choices=["standard", "hd"], help='The quality of the DALL-E image. Default is "standard".')

    args = parser.parse_args()
    assert args.art_model in [model.value for model in SupportedModels], f"Art model {args.art_model} is not supported. Supported models are {[model.value for model in SupportedModels]}"

    foundry = FblthpFoundries(args)
    foundry.run()
