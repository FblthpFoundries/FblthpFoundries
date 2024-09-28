import json
import os
import sys
import re
os.environ["HEADLESS"] = "True"
from constants import PROXYSHOP_PATH, TEMPLATE_PATH

sys.path.append(PROXYSHOP_PATH)
from pathlib import Path
from src.layouts import NormalLayout
from src.templates import BorderlessVectorTemplate

class BaseRenderer():
    def __init__(self):
        pass
    def render_card(self, card, art_path):
        pass

class ProxyshopRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
    def scryfall_markup(card):
        output = {}
        output["object"] = "card"
        output["name"] = card.name
        output["lang"] = "en"
        output["layout"] = "normal" #TODO: Fix planeswalkers since they will probably break here
        output["mana_cost"] = card.mana_cost
        output["type_line"] = card.type_line
        output["oracle_text"] = card.oracle_text
        if "power" in input:
            output["power"] = card.power
        if "toughness" in input:
            output["toughness"] = card.toughness
        if "loyalty" in input:
            output["loyalty"] = card.loyalty
        output["collector_number"] = "42"
        if "rarity" in input:
            output["rarity"] = card.rarity.lower()
        else:
            output["rarity"] = "rare"
        output["flavor_text"] = card.flavor_text
        output["artist"] = "Fblthp Foundries"
        output["set"] = "war" #TODO: Change this to something cooler

        return output
    def render_card(self, card, art_path):
        # # Initialize the Photoshop application handler
        # photoshop_app = PhotoshopHandler()

        # Define the path to your template PSD file
        template_path = Path(TEMPLATE_PATH)  # Adjust the path as needed

        # Define the Scryfall data for your card
        scryfall_data = self.scryfall_markup(card)

        print(scryfall_data)

        # Load the path to your card's image file
        art_file_path = Path(art_path)  # Adjust the path as needed



        # Instantiate the layout object
        card_layout = NormalLayout(
            scryfall=scryfall_data,
            file={
                'file': art_file_path,
                'name': scryfall_data['name'],
                'artist': scryfall_data['artist'],
                'set': scryfall_data['set'],
                'creator': None
            }
        )
        card_layout.template_file = template_path
        # Instantiate the template directly
        template_object = BorderlessVectorTemplate(card_layout)

        # Start the rendering process

        # Assuming the template object has an execute method
        current_render = template_object  # Directly use the instantiated template object
        result = current_render.execute()

        if result:
            print(f"Rendering completed successfully!")
        else:
            print(f"Rendering failed: Unknown error")

    def render_folder(self, folder_path):
        folder = Path(folder_path)
        
        # Find all .json and .txt files
        json_files = list(folder.glob("*.json")) + list(folder.glob("*.txt"))

        for json_file in json_files:
            art_file = json_file.with_suffix('.png')
            if art_file.exists():
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                self.render_card(json_data, art_file)
            else:
                print(f"Art file not found for {json_file.stem}. Skipping...")



if __name__ == "__main__":
    import argparse

    #parser = argparse.ArgumentParser(description="Render all cards in a folder.")
    #parser.add_argument("folder", help="Path to the folder containing card .json and .png files", required=False)
    #args = parser.parse_args()
    #print(args.folder)

    card = {"flavor_text":"In the face of overwhelming odds, goblin shamans always succeed.","loyalty":"","mana_cost":"<4> <R>","name":"Goblin Looter","oracle_text":"Goblin Looter enters with two oil counters on it. \n <T>: Goblin Looter gains flying until end of turn.","power":"4","toughness":"4","type_line":"Creature - Goblin Rogue"}

    renderer = ProxyshopRenderer()
    renderer.render_card(card, PROXYSHOP_PATH + '\\..\\stormCrow.jpg')
    
    #render_folder(args.folder)
    # fd = "C:\\Users\\Sam\\Documents\\FblthpFoundries\\fblthp\\art\\out\\run"
    # for i in range(1, 14):
    #     render_folder(fd + str(i))
    