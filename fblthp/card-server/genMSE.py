from zipfile import ZipFile, ZIP_DEFLATED
import os, logging
from datetime import datetime
from pathlib import Path
from card import Card
BASE_DIR = Path(__file__).resolve().parent

preamble = """mse_version: 2.0.2
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
	magic-m15-mainframe-planeswalker:
		text_box_mana_symbols: magic-mana-small.mse-symbol-font
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
\tcard_code_text: 
\timage_2: 
\tmainframe_image: 
\tmainframe_image_2:
"""

def createMSE(cards: list[Card], name: str = 'set', logger : logging.Logger = None) -> str:
    now = datetime.now()
    date = now.strftime('%Y-%m-%d %H:%M:%S')
    mse = preamble
    for card in cards:
        mse += card.genMSE()+ fluff.replace("DATE", date)

    mse += end

    with open(f'{BASE_DIR}/toZip/set', 'w') as f:
        f.write(mse)

    with ZipFile(f'{BASE_DIR}/{name}.mse-set', 'w', ZIP_DEFLATED) as zip:
        zip.write(f'{BASE_DIR}/toZip/set','set')
        zip.write( f'{BASE_DIR}/toZip/fblthpAI.mse-symbol', 'fblthpAI.mse-symbol')
        zip.write(f'{BASE_DIR}/images/default/gradient.png', 'gradient')

        for card in cards:
            if card.image_path:
                zip.write(card.image_path, card.name.replace(' ',''))

    os.remove(BASE_DIR/"toZip"/"set")

    return f'{BASE_DIR}/{name}.mse-set'
