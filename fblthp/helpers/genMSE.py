from .magicCard import Card
import os
from zipfile import ZipFile
from datetime import datetime
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR 

preamble = """
mse_version: 2.0.2
game: magic
game_version: 2020-04-25
stylesheet: m15-altered
stylesheet_version: 2024-01-05
set_info:
	symbol: fblthpAI.mse-symbol
	masterpiece_symbol: 
	mana_symbol_options: enable in casting costs, enable in text boxes, colored mana symbols, hybrid with colors
styling:
	magic-m15-altered:
		text_box_mana_symbols: magic-mana-small.mse-symbol-font
		level_mana_symbols: magic-mana-large.mse-symbol-font
		overlay:
"""

end = """
version_control:
	type: none
apprentice_code:
"""

fluff = """
\thas_styling: false
\tnotes: 
\ttime_created: DATE
\ttime_modified: DATE
\timage: 
\tcard_code_text: 
\timage_2: 
\tmainframe_image: 
\tmainframe_image_2:
"""

def createMSE(name, cards):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d %H:%M:%S')
    mse = preamble
    for card in cards:
        types = card.type.split("-")
        angry = card.oracle.replace('\n', '\n\t\t')
        oracle = card.oracle if not '\n' in card.oracle else f"\n\t\t{angry}"
        flavor = card.flavor
        text = 'card:\n'
        text += f'\tname: {card.name}\n'
        text += f'\trule_text: {oracle.replace("} {", "").replace("{","").replace("}", "")}\n'
        text += f'\tsuper_type: {types[0]}\n'
        if len(types) > 1:
            text += f'\tsub_type: {types[1]}\n'
        if card.power:
            text += f'\tpower: {card.power}\n'
        if card.toughness:
            text += f'\ttoughness: {card.toughness}\n'
        if card.loyalty:
            text += f'\tloyalty: {card.loyalty}\n'
        if card.flavor:
            print('flavortown')
            text += f'\tflavor_text: {flavor}\n'
        if card.mc:
            text += f'\tcasting_cost: {card.mc.replace("} {", "").replace("{","").replace("}", "")}\n'

        mse += text[:-1] + fluff.replace("DATE", date)

    mse += end

    with open(f'{BASE_DIR}/helpers/toZip/set', 'w') as f:
        f.write(mse)

    with ZipFile(f'{BASE_DIR}/{name}.mse-set', 'w') as zip:
        zip.write(f'{BASE_DIR}/helpers/toZip/set')
        zip.write(f'{BASE_DIR}/helpers/toZip/fblthpAI.mse-symbol')