import json
import os
import sys
import re
os.environ["HEADLESS"] = "True"
from constants import PROXYSHOP_PATH, TEMPLATE_PATH

sys.path.append(PROXYSHOP_PATH)
from pathlib import Path
from Proxyshop.src.layouts import NormalLayout
from Proxyshop.src.templates import BorderlessVectorTemplate
from Proxyshop.src.utils.adobe import PhotoshopHandler

def adjust_json(input):
    output = {}
    output["object"] = "card"
    output["name"] = input["name"]
    output["lang"] = "en"
    output["layout"] = "normal" #TODO: Fix planeswalkers since they will probably break here
    output["mana_cost"] = input["mana_cost"].replace("<", "{").replace(">", "}")
    output["type_line"] = input["type_line"]
    output["oracle_text"] = re.sub(r'\n+', '\n',input["oracle_text"].replace("<", "{").replace(">", "}").replace("\\n", "\n")).replace("*", "").replace("T:", "{T}:").replace("T,","{T},")
    if "power" in input:
        output["power"] = input["power"]
    if "toughness" in input:
        output["toughness"] = input["toughness"]
    if "loyalty" in input:
        output["loyalty"] = input["loyalty"]
    output["collector_number"] = "42"
    if "rarity" in input:
        output["rarity"] = input["rarity"].lower()
    else:
        output["rarity"] = "rare"
    output["flavor_text"] = input["flavor_text"]
    if "image_generator" in input:
        output["artist"] = input["image_generator"]
    else:
        output["artist"] = "DALL-E"
    output["set"] = "war" #TODO: Change this to something cooler

    return output


def render_card(path_to_card, path_to_art):
    # # Initialize the Photoshop application handler
    # photoshop_app = PhotoshopHandler()

    # Define the path to your template PSD file
    template_path = Path(TEMPLATE_PATH)  # Adjust the path as needed

    # Define the Scryfall data for your card
    scryfall_data = adjust_json(path_to_card)

    # Load the path to your card's image file
    art_file_path = Path(path_to_art)  # Adjust the path as needed



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
        pass # print(f"Rendering completed successfully!")
    else:
        print(f"Rendering failed: Unknown error")

def render_folder(folder_path):
    folder = Path(folder_path)
    
    # Find all .json and .txt files
    json_files = list(folder.glob("*.json")) + list(folder.glob("*.txt"))

    for json_file in json_files:
        art_file = json_file.with_suffix('.png')
        if art_file.exists():
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            render_card(json_data, art_file)
        else:
            print(f"Art file not found for {json_file.stem}. Skipping...")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render all cards in a folder.")
    parser.add_argument("folder", help="Path to the folder containing card .json and .png files")
    args = parser.parse_args()
    print(args.folder)
    render_folder(args.folder)
    # fd = "C:\\Users\\Sam\\Documents\\FblthpFoundries\\fblthp\\art\\out\\run"
    # for i in range(1, 14):
    #     render_folder(fd + str(i))
    