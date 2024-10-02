import json, os, sys, re, subprocess
from .magicCard import Card
from .genMSE import createMSE

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
FBLTHP_OUT = BASE_DIR / "images" / "rendered"



class BaseRenderer():
    def __init__(self):
        if not os.path.exists(FBLTHP_OUT):
            os.makedirs(FBLTHP_OUT)
        pass
    def render_cards(self, cards:list[Card]):
        pass

class MSERenderer(BaseRenderer):
    def __init__(self):
        super().__init__()

    def render_cards(self, cards:list[Card]):
        MSE_PATH = BASE_DIR / 'Basic-M15-Magic-Pack'
        print(len(cards))

        zipPath = createMSE('tmp', cards)

        renderScript = f'for each c in set.cards do write_image_file(c, file: c.name + \".png\")'
            

        scriptProcess = subprocess.Popen(['echo', renderScript], stdout=subprocess.PIPE, text=True)
        renderProcess = subprocess.Popen([MSE_PATH/'mse', '--cli', zipPath], stdin=scriptProcess.stdout, stdout= subprocess.PIPE, text=True)

        output, error = renderProcess.communicate()

        #print(output)
        #print(error)
        for card in cards:
            fileName = card.name[1:] if card.name[0] == ' ' else card.name
            os.rename(BASE_DIR/f'{fileName}.png', FBLTHP_OUT/f'{fileName}.png')
            card.render_path = FBLTHP_OUT/f'{fileName}.png'

        os.remove(BASE_DIR/'tmp.mse-set')
        for file in os.listdir(BASE_DIR):
            if file.endswith('.png'):
                os.remove(file)


class ProxyshopRenderer(BaseRenderer):
    def __init__(self):
        
        super().__init__()
    def scryfall_markup(self, card):
        output = {}
        output["object"] = "card"
        output["name"] = card.name
        output["lang"] = "en"
        output["layout"] = "normal" #TODO: Fix planeswalkers since they will probably break here
        print(card.mana_cost)
        output["mana_cost"] = card.mana_cost.replace(" ", "")
        output["type_line"] = card.type_line
        output["oracle_text"] = re.sub(r'\n+', '\n', card.oracle_text)
        if card.power:
            output["power"] = card.power
        if card.toughness:
            output["toughness"] = card.toughness
        if card.loyalty:
            output["loyalty"] = card.loyalty
        output["collector_number"] = "42"
        if card.rarity:
            output["rarity"] = card.rarity.lower()
        else:
            output["rarity"] = "rare"
        output["flavor_text"] = card.flavor_text
        output["artist"] = "Fblthp Foundries"
        output["set"] = "ll" #TODO: Change this to something cooler

        return output
    
    def render_cards(self, cards: list[Card]):
        for card in cards:
            card.render_path = self.render_card(card, card.image_path)

    def render_card(self, card, art_path):
        os.environ["HEADLESS"] = "True"
        PROXYSHOP_PATH = BASE_DIR / "Proxyshop"

        if str(PROXYSHOP_PATH) not in sys.path:
            sys.path.append(str(PROXYSHOP_PATH))

        # Assuming 'src' is inside the Proxyshop folder or another relative path
        SRC_PATH = PROXYSHOP_PATH / 'src'

        # Add src to sys.path if it's not already there
        if str(SRC_PATH) not in sys.path:
            sys.path.append(str(SRC_PATH))

        from layouts import NormalLayout
        from templates import BorderlessVectorTemplate
        # # Initialize the Photoshop application handler
        # photoshop_app = PhotoshopHandler()

        # Define the path to your template PSD file
        template_path = Path(BASE_DIR / "helpers" / "borderless-vector.psd")  # Adjust the path as needed

        # Define the Scryfall data for your card
        scryfall_data = self.scryfall_markup(card)

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
        card_layout.symbol_svg = BASE_DIR / "images" / "defaults" / f"{card.rarity}.svg"
        #card_layout.symbol_svg = BASE_DIR / "images" / "defaults" / f"supreme rare.svg"
        
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
        PROXYSHOP_OUT = BASE_DIR / "Proxyshop" / "out"
        for filename in os.listdir(PROXYSHOP_OUT):
                if filename.startswith(card.name):
                    new_path = os.path.join(FBLTHP_OUT, f"{card.name}_{card.uuid}.png")
                    os.rename(os.path.join(PROXYSHOP_OUT, filename), new_path)
                    break
        return new_path

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
    import cv2

    card = {"flavor_text":"In the face of overwhelming odds, goblin shamans always succeed.","loyalty":"","mana_cost":"<4> <R>","name":"Goblin Looter","oracle_text":"Goblin Looter enters with two oil counters on it. \n <T>: Goblin Looter gains flying until end of turn.","power":"4","toughness":"4","type_line":"Creature - Goblin Rogue"}
    card = Card(card)
    renderer = MSERenderer()
    print(BASE_DIR)
    cv2.imshow("test", cv2.imread(BASE_DIR / Path("stormCrow.jpg")))
    renderer.render_card(card, BASE_DIR / Path("stormCrow.jpg"))