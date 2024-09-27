import json
import random
import openai
import requests
import re
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import uuid
from PyQt6.QtCore import QObject, pyqtSignal
from openai import OpenAI, BadRequestError
from concurrent.futures import ThreadPoolExecutor, as_completed
from .distribution import seed_card
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "helpers" / "prompts"
IMAGE_DIR = BASE_DIR / "images" / "downloaded"
NONSENSE_DIR = BASE_DIR / "nonsense"
# CARD GENERATORS

class BaseCardGenerator():
    def __init__(self):
        pass
    def create_cards(self, number, update_function=None):
        '''
        Creates and returns a list of {number} cards in dictionary form
        '''
        pass
    def reroll(self):
        pass
    def load_card_settings(self, path):
        pass
    
class ChatGPTCardGenerator(BaseCardGenerator):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__()
        from .constants import API_KEY
        openai.api_key = API_KEY
        self.model = "gpt-4o-mini"
        self.cube_yml = PROMPTS_DIR / "defaults.yml"

        self.update_card_settings()

    def create_cards(self, number, update_function=None):
        self.update_card_settings()

        cards = []
        i = 0
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.generate_card_gpt) for _ in range(number)]

            for future in as_completed(futures):
                success, card = future.result()
                if success:
                    cards.append(card)
                    i += 1
                    if update_function:
                        update_function(i/number)
        return cards

    def reroll(self):
        success = False
        while not success:
            success, card = self.generate_card_gpt()
        return card
        
    def load_card_settings(self, path):
        self.cube_yml = path

    def update_card_settings(self):
        if self.cube_yml:
            import yaml
            with open(self.cube_yml, 'r') as f:
                data = yaml.safe_load(f)

    def generate_card_gpt(self):
        

        card_type, colors, mana_cost, rarity = seed_card()

        color_identity = ", ".join(colors) if colors else "Colorless"
        subbie = ""
        if "Artifact" in card_type:
            if "Vehicle" not in card_type:
                subbie = f"""{('- The card is a mana rock.' if random.random() < 0.4 
                else ("- The card has an ability with {T} as part of the cost." if random.random() < 0.5 
                else "")) if 'Artifact' in card_type else ''}"""
            else:
                subbie = "The card is a vehicle"
        # Generate card 'theme'
        with open(PROMPTS_DIR / "themes.txt", "r") as f:
            theme_prompt = f.read().format(
                color_identity=color_identity,
                card_type=card_type,
                mana_cost=mana_cost,
                rarity=rarity,
                subbie=subbie
            )
        success, themes = self.ask_gpt(theme_prompt)
        if not success:
            raise Exception("Failed to generate themes using GPT")
        themes = themes['themes']
        theme = random.choice(themes)

        # Generate Card Name
        with open(PROMPTS_DIR / "names.txt", "r") as f:
            name_prompt = f.read().format(
                color_identity=color_identity,
                card_type=card_type,
                mana_cost=mana_cost,
                rarity=rarity,
                subbie=subbie,
                theme=theme
            )
        success, names = self.ask_gpt(name_prompt)

        if not success:
            raise Exception("Failed to generate names using GPT")
        names = names['names']
        name = random.choice(names)

        
        adds = self.additions(card_type, rarity)
        with open(PROMPTS_DIR / "card.txt", "r") as f:
            card_prompt = f.read().format(
                color_identity=color_identity,
                card_type=card_type,
                mana_cost=mana_cost,
                rarity=rarity,
                subbie=subbie,
                theme=theme,
                name=name,
                additional_specs=adds
            )
        success, initial_card = self.ask_gpt(card_prompt)
        if not success:
            raise Exception("Failed to generate card using GPT")

        with open(PROMPTS_DIR / "balancing.txt", "r") as f:
            balancing_prompt = f.read().format(
                initial_card=initial_card
            )
        
        success, new_card = self.ask_gpt(balancing_prompt)
        if not success:
            raise Exception("Failed to balance card using GPT")
        card = initial_card.copy()
        card['oracle_text'] = new_card['oracle_text']
        card['adjustment'] = new_card['adjustment']
        card['theme'] = theme
        #card['themes'] = themes
        #card['names'] = names
        card['initial-card'] = initial_card
        card['chatgpt-prompt'] = card_prompt

        return True, card

    # Helper function to add additional prompting when generating a card
    def additions(self, card_type, rarity):
        strength_dict = {
            "common": "somewhat weak and have only one ability",
            "uncommon": "reasonably balanced with good synergy",
            "rare": "strong, with synergistic abilities",
            "mythic rare": "a strong, high-impact card with multiple syngergistic abilities"
        }
        adds = []

        if 'Creature' in card_type:
            adds.append('The card should have creature subtypes fitting its theme.')
        if 'Creature' not in card_type and 'Vehicle' not in card_type:
            adds.append('The card should have empty fields for power and toughness.')
        if 'Planeswalker' in card_type:
            adds.append('Planeswalkers should have a loyalty value in the "loyalty" field, as well as abilities that reflect their character. Planeswalkers also have a subtype with their first name in the type line. i.e. Planeswalker - <firstname>')
        if 'Land' in card_type:
            adds.append('Lands should have no mana cost, and have at least one ability that produces mana.')
        if "Vehicle" in card_type:
            adds.append("The card must have a crew ability with a cost.")
        if random.random() < 0.3:
            adds.append("Give the card a new keyword ability.")

        adds.append(f"Make it {strength_dict[rarity.lower()]}")
        return ("-" + "-\n".join(adds) + "\n") if adds else ""

    # Helper function to interact with the GPT model (make sure prompts ask for JSON return)
    def ask_gpt(self, prompt, header="You are an expert in generating Magic: The Gathering cards."):
        #print(f"ME: {prompt}")
        
        
        response = openai.chat.completions.create(
            model=self.model,
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
        #print(f"CHATGPT: {resp}")
        try:
            resp = json.loads(resp)
            return True, resp
        except:
            return False, None

class LocalCardGenerator(BaseCardGenerator):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from .secret_tokens import hugging_face_token
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        print(f"Loaded tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained('FblthpAI/magic_model', token = hugging_face_token())
        print(f"Loaded model ({self.get_model_size_in_megabytes()}MB)")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.model.to(self.device)
        self.cards = []
        self.batchSize = 10
    def create_cards(self, number, updateFunc = None):
        #fills up card queue untill it has enough to serve request

        while len(self.cards) < number:
            self._generate(lambda n : updateFunc(n/number))

        toReturn = self.cards[:number]
        self.cards = self.cards[number:]

        return toReturn
    def get_model_size_in_megabytes(self):
        # Get the number of parameters in the model
        param_size = sum(p.numel() for p in self.model.parameters())

        # Each parameter typically takes 4 bytes (float32)
        param_size_in_bytes = param_size * 4

        # Convert from bytes to megabytes
        param_size_in_megabytes = param_size_in_bytes / (1024 ** 2)
        return int(param_size_in_megabytes)
    def reroll(self):
        if len(self.cards) > 0:
            return self.cards.pop()

        self._generate()        

        return self.cards.pop()

    def _generate(self, updateFunc = None):
        text = ['<tl>'] * self.batchSize
        encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model.generate(
            **encoded_input,
            do_sample=True,
            temperature = 0.9,
            max_length =400,
        )

        cardOutput = self.tokenizer.batch_decode(output)
        cardTokens = ''
        #collect all but last card from each batch
        #last card often is cut off by max_length
        for batch in cardOutput:
            for card in batch.split('<eos>')[:-1]:
                cardTokens += card.replace('\u2212', '-').replace('\u2014', '-').replace('\u2022', '')

        cardTokens = self._parse_card_data(cardTokens)

        print(len(cardTokens))

        for card in cardTokens:
            if not self._is_valid_card(card):
                pass
            self.cards.append(card)
            if updateFunc:
                updateFunc(len(self.cards))
    def _is_valid_card(self,card):
        if not 'name' in card:
            return False
        if not 'oracle_text' in card:
            return False
        if not 'type_line' in card:
            return False
        return True
    
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


# IMAGE GENERATORS


class BaseImageGenerator(QObject):
    def __init__(self):
        super().__init__()
    def generate_image(self, card):
        pass

class DALLEImageGenerator(BaseImageGenerator):
    def __init__(self, quality="standard", size="1024x1024", additional_prompt="", text_model="gpt-4o-mini"):
        super().__init__()
        from .constants import API_KEY
        openai.api_key = API_KEY
        self.quality = quality
        self.size = size
        self.additional_prompt=""
        self.text_model = text_model
    def generate_image(self, card):
        success, image_prompt = self.generate_image_prompt(card)
        if success:
            success, image_data = self.generate_an_image(image_prompt)
            if success:
                image = Image.open(BytesIO(image_data))

                # Ensure the image directory exists
                if not os.path.exists(IMAGE_DIR):
                    os.mkdir(IMAGE_DIR)

                # Define the full path to save the image
                image_path = os.path.join(IMAGE_DIR, f"{card.name}_{uuid.uuid4()}.png")
                image.save(image_path)

                return image_path

            else:
                return None
    def generate_images(self, cards):
        images = []
        for card in cards:
            success, image_prompt = self.generate_image_prompt(card)
            if success:
                success, image_data = self.generate_an_image(image_prompt)
                if success:
                    images.append(image_data)
                else:
                    images.append(None)
            else:
                images.append(None)
        return images
    
    def generate_image_prompt(self, card, model="gpt-4o-mini", additional_prompt="", max_chars=800):
        # Send the request to ChatGPT
        with open(PROMPTS_DIR / "EXTRACTION.txt", "r") as f:
            prompt = f.read().format(
                type_line=card.type_line,
                name=card.name,
                mana_cost=card.mana_cost,
                oracle_text=card.oracle_text,
                power=card.power,
                toughness=card.toughness,
                loyalty=card.loyalty,
                flavor_text=card.flavor_text,
                rarity=card.rarity,
                theme=card.theme,
                code_adds = "",
                additional_prompt=self.additional_prompt,
                max_chars=max_chars
            )
        success, out = self.ask_gpt(prompt)
        image_prompt = out["prompt"]
        return True, image_prompt
    def generate_an_image(self, prompt):

        client = OpenAI(api_key=openai.api_key)
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                n=1
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            return True, image_data
        except BadRequestError as e:
            print(f"Image rejected by safety system: {e}")
            return False, None
        except Exception as e:
            print(f"Failed to generate image using DALL-E: {e}")
            return False, None
    # Helper function to interact with the GPT model (make sure prompts ask for JSON return)
    def ask_gpt(self, prompt, header="You are an expert in generating Magic: The Gathering cards."):
        #print(f"ME: {prompt}")
        
        
        response = openai.chat.completions.create(
            model=self.text_model,
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
        #print(f"CHATGPT: {resp}")
        return True, json.loads(resp)

class SD3ImageGenerator(BaseImageGenerator):
    def __init__(self):
        super().__init__()
        self.pipe = None

    def __load(self):
        from diffusers import StableDiffusion3Pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16
            )
        self.pipe.to('cuda')
        self.pipe.enable_cpu_offload()


    def generate_images(self, cards):
        pass
    def generate_image(self, card):
        if not self.pipe:
            self.__load()
        with open(PROMPTS_DIR / "SD3prompt.txt",'r') as f:
            prompt = f.read().format(
                type_line=card.type_line,
                name=card.name,
                mana_cost=card.mana_cost,
                flavor_text=card.flavor_text,
            )
        try:
            print('starting')

            img = self.pipe(
                prompt=prompt,
                num_images_per_prompt=1,
                num_inference_steps=15,
                height=800,
                width=1024
            ).images[0]
            print('done')
            print(type(img))


            # Ensure the image directory exists
            if not os.path.exists(IMAGE_DIR):
                os.mkdir(IMAGE_DIR)

            # Define the full path to save the image
            image_path = os.path.join(IMAGE_DIR, f"{card.name}_{uuid.uuid4()}.png")
            img.save(image_path)

            return image_path
        except Exception as e:
            print(f"Failed to generate image locally: {e}")
            raise

class FluxImageGenerator(BaseImageGenerator):
    def __init__(self):
        super().__init__()
    def generate_images(self, cards):
        pass

class PixabayImageGenerator(BaseImageGenerator):
    def __init__(self):
        super().__init__()
        from helpers.constants import PIXABAY_API_KEY
        self.api_key = PIXABAY_API_KEY
        self.search_url = "https://pixabay.com/api/"

    def generate_image(self, card):
        # Search for an image matching card.name (or fantasy-related keywords)
        params = {
            "key": self.api_key,
            "q": card.name,  # You can add custom tags like "fantasy" if needed
            "image_type": "illustration",
            "category": "fantasy",
            #"safesearch": "true",
            "per_page": 3
        }
        response = requests.get(self.search_url, params=params)
        response.raise_for_status()

        results = response.json()
        if not results['hits']:
            print(f"No images found for {card.name}")
            return None

        # Get the image URL
        image_url = results['hits'][0]['largeImageURL']
        
        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        # Load the image into PIL and resize it
        image = Image.open(BytesIO(image_response.content))
        image = image.resize((800, 600))

        # Ensure the image directory exists
        if not os.path.exists(IMAGE_DIR):
            os.mkdir(IMAGE_DIR)

        # Define the full path to save the image
        image_path = os.path.join(IMAGE_DIR, f"{card.name}_{card.uuid}.png")
        image.save(image_path)

        return image_path
class TestImageGenerator(BaseImageGenerator):
    def __init__(self):
        super().__init__()
    def generate_image(self, card):
        # Define the fixed size of the image
        width, height = 800, 600

        # Define the gradient colors (Red to Blue)
        color1 = (255, 0, 0)  # Red
        color2 = (0, 0, 255)  # Blue

        # Create a new blank image (RGB mode)
        image = Image.new("RGB", (width, height))

        # Get pixel map
        pixels = image.load()

        # Generate the gradient
        for x in range(width):
            for y in range(height):
                # Calculate the interpolation factor (between 0 and 1)
                factor = x / width

                # Interpolate the color components (R, G, B)
                r = int(color1[0] * (1 - factor) + color2[0] * factor)
                g = int(color1[1] * (1 - factor) + color2[1] * factor)
                b = int(color1[2] * (1 - factor) + color2[2] * factor)

                # Set the pixel color
                pixels[x, y] = (r, g, b)
        if not os.path.exists(IMAGE_DIR):
            os.mkdir(IMAGE_DIR)

        # Define the full path to save the image
        image_path = os.path.join(IMAGE_DIR, "gradient.png")

        # Save the image
        image.save(image_path)
        return image_path
        # Return the path to the saved image
if __name__ == "__main__":
    import cv2
    cardgenerator = ChatGPTCardGenerator()
    cards = cardgenerator.create_cards(1)
    #print(" PRINTING CARDS!!!!!")
    #[print(c) for c in cards]
    imagegenerator = DALLEImageGenerator()
    images = imagegenerator.generate_images(cards)
    for img in images:
        if img:
            nparr = np.frombuffer(img, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow("Image", img_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()







