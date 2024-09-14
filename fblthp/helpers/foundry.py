import json
import random
import openai
import requests
import re
import torch
import numpy as np

from openai import OpenAI, BadRequestError
from distribution import seed_card
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "helpers" / "prompts"

# CARD GENERATORS

class BaseCardGenerator():
    def __init__(self):
        pass
    def create_cards(self, number):
        '''
        Creates and returns a list of {number} cards in dictionary form
        '''
        pass
    
class ChatGPTCardGenerator(BaseCardGenerator):
    def __init__(self, model="gpt-4o-mini"):
        super().__init__()
        self.model = model
    
    def create_cards(self, number):
        cards = []
        i = 0
        while i < number:
            success, card = self.generate_card_gpt()
            if success:
                cards.append(card)
                i += 1
        return cards

    
    def generate_card_gpt(self):
        from constants import API_KEY
        openai.api_key = API_KEY

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
        card['themes'] = themes
        card['names'] = names
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
        print(f"ME: {prompt}")
        
        
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
        print(f"CHATGPT: {resp}")
        return True, json.loads(resp)

class LocalCardGenerator(BaseCardGenerator):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
    def create_cards(self, number):
        pass
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
    def parse_local_card(input_text):
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


# IMAGE GENERATORS


class BaseImageGenerator():
    def __init__(self):
        pass
    def generate_images(self, cards):
        pass

class DALLEImageGenerator(BaseImageGenerator):
    def __init__(self, quality="standard", size="1024x1024", additional_prompt="", text_model="gpt-4o-mini"):
        super().__init__()
        from constants import API_KEY
        openai.api_key = API_KEY
        self.quality = quality
        self.size = size
        self.additional_prompt=""
        self.text_model = text_model
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
                type_line=card['type_line'],
                name=card['name'],
                mana_cost=card['mana_cost'],
                oracle_text=card['oracle_text'],
                power=card['power'],
                toughness=card['toughness'],
                loyalty=card['loyalty'],
                flavor_text=card['flavor_text'],
                rarity=card['rarity'],
                theme=card['theme'],
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
        print(f"ME: {prompt}")
        
        
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
        print(f"CHATGPT: {resp}")
        return True, json.loads(resp)

class SD3ImageGenerator(BaseImageGenerator):
    def __init__(self):
        super().__init__()
    def generate_images(self, cards):
        pass
    def generate_an_image(self, prompt, model="SD3"):
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

class FluxImageGenerator(BaseImageGenerator):
    def __init__(self):
        super().__init__()
    def generate_images(self, cards):
        pass

class GoogleImageGenerator(BaseImageGenerator):
    def __init__(self):
        super().__init__()
    def generate_images(self, cards):
        pass


if __name__ == "__main__":
    import cv2
    cardgenerator = ChatGPTCardGenerator()
    cards = cardgenerator.create_cards(1)
    print(" PRINTING CARDS!!!!!")
    [print(c) for c in cards]
    imagegenerator = DALLEImageGenerator()
    images = imagegenerator.generate_images(cards)
    for img in images:
        if img:
            nparr = np.frombuffer(img, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow("Image", img_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()






