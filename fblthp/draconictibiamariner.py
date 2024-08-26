from constants import PROXYSHOP_PATH

import sys
import os
import json
import re
from typing import Union, Optional, Callable
sys.path.append(PROXYSHOP_PATH)


# Standard Library Imports
from pathlib import Path

# Third Party Imports
import click

# Local Imports
from src import CON, TEMPLATE_DEFAULTS
from src._loader import TemplateDetails
from src.cards import CardDetails
from src.enums.mtg import LayoutType, LayoutScryfall, CardTypes, CardTypesSuper
from src.layouts import layout_map

from src.templates import BaseTemplate
from src.templates.normal import BorderlessVectorTemplate
from src.utils.files import load_data_file
from src.layouts import (
    CardLayout,
    layout_map,
    assign_layout,
    join_dual_card_layouts)
from src.layouts import NormalLayout
from src import (
        APP, CFG, CON, CONSOLE, ENV,
        PLUGINS, TEMPLATES, TEMPLATE_MAP, TEMPLATE_DEFAULTS)

if __name__ == "__main__":
    card = json.loads('''{"object":"card","id":"036ef8c9-72ac-46ce-af07-83b79d736538","oracle_id":"000d5588-5a4c-434e-988d-396632ade42c","multiverse_ids":[83282],"mtgo_id":22609,"mtgo_foil_id":22610,"tcgplayer_id":12835,"cardmarket_id":12551,"name":"Storm Crow","lang":"en","released_at":"2005-07-29","uri":"https://api.scryfall.com/cards/036ef8c9-72ac-46ce-af07-83b79d736538","scryfall_uri":"https://scryfall.com/card/9ed/100/storm-crow?utm_source=api","layout":"normal","highres_image":true,"image_status":"highres_scan","image_uris":{"small":"https://cards.scryfall.io/small/front/0/3/036ef8c9-72ac-46ce-af07-83b79d736538.jpg?1562730661","normal":"https://cards.scryfall.io/normal/front/0/3/036ef8c9-72ac-46ce-af07-83b79d736538.jpg?1562730661","large":"https://cards.scryfall.io/large/front/0/3/036ef8c9-72ac-46ce-af07-83b79d736538.jpg?1562730661","png":"https://cards.scryfall.io/png/front/0/3/036ef8c9-72ac-46ce-af07-83b79d736538.png?1562730661","art_crop":"https://cards.scryfall.io/art_crop/front/0/3/036ef8c9-72ac-46ce-af07-83b79d736538.jpg?1562730661","border_crop":"https://cards.scryfall.io/border_crop/front/0/3/036ef8c9-72ac-46ce-af07-83b79d736538.jpg?1562730661"},"mana_cost":"{1}{U}","cmc":2,"type_line":"Creature — Bird","oracle_text":"Flying (This creature can't be blocked except by creatures with flying or reach.)","power":"1","toughness":"2","colors":["U"],"color_identity":["U"],"keywords":["Flying"],"legalities":{"standard":"not_legal","future":"not_legal","historic":"not_legal","timeless":"not_legal","gladiator":"not_legal","pioneer":"not_legal","explorer":"not_legal","modern":"legal","legacy":"legal","pauper":"legal","vintage":"legal","penny":"legal","commander":"legal","oathbreaker":"legal","standardbrawl":"not_legal","brawl":"not_legal","alchemy":"not_legal","paupercommander":"legal","duel":"legal","oldschool":"not_legal","premodern":"legal","predh":"legal"},"games":["paper","mtgo"],"reserved":false,"foil":false,"nonfoil":true,"finishes":["nonfoil"],"oversized":false,"promo":false,"reprint":true,"variation":false,"set_id":"e70c8572-4732-4e92-a140-b4e3c1c84c93","set":"9ed","set_name":"Ninth Edition","set_type":"core","set_uri":"https://api.scryfall.com/sets/e70c8572-4732-4e92-a140-b4e3c1c84c93","set_search_uri":"https://api.scryfall.com/cards/search?order=set&q=e%3A9ed&unique=prints","scryfall_set_uri":"https://scryfall.com/sets/9ed?utm_source=api","rulings_uri":"https://api.scryfall.com/cards/036ef8c9-72ac-46ce-af07-83b79d736538/rulings","prints_search_uri":"https://api.scryfall.com/cards/search?order=released&q=oracleid%3A000d5588-5a4c-434e-988d-396632ade42c&unique=prints","collector_number":"100","digital":false,"rarity":"common","flavor_text":"Storm crow descending, winter unending. Storm crow departing, summer is starting.","card_back_id":"0aeebaf5-8c7d-4636-9e82-8c27447861f7","artist":"John Matson","artist_ids":["a1685587-4b55-446b-b420-c37b202ed3f1"],"illustration_id":"d01aa92b-0582-4e1e-a7b0-737b2ad4e462","border_color":"white","frame":"2003","full_art":false,"textless":false,"booster":true,"story_spotlight":false,"edhrec_rank":17354,"penny_rank":12596,"prices":{"usd":"0.23","usd_foil":null,"usd_etched":null,"eur":"0.04","eur_foil":null,"tix":"0.04"},"related_uris":{"gatherer":"https://gatherer.wizards.com/Pages/Card/Details.aspx?multiverseid=83282&printed=false","tcgplayer_infinite_articles":"https://tcgplayer.pxf.io/c/4931599/1830156/21018?subId1=api&trafcat=infinite&u=https%3A%2F%2Finfinite.tcgplayer.com%2Fsearch%3FcontentMode%3Darticle%26game%3Dmagic%26partner%3Dscryfall%26q%3DStorm%2BCrow","tcgplayer_infinite_decks":"https://tcgplayer.pxf.io/c/4931599/1830156/21018?subId1=api&trafcat=infinite&u=https%3A%2F%2Finfinite.tcgplayer.com%2Fsearch%3FcontentMode%3Ddeck%26game%3Dmagic%26partner%3Dscryfall%26q%3DStorm%2BCrow","edhrec":"https://edhrec.com/route/?cc=Storm+Crow"},"purchase_uris":{"tcgplayer":"https://tcgplayer.pxf.io/c/4931599/1830156/21018?subId1=api&u=https%3A%2F%2Fwww.tcgplayer.com%2Fproduct%2F12835%3Fpage%3D1","cardmarket":"https://www.cardmarket.com/en/Magic/Products/Singles/Ninth-Edition/Storm-Crow?referrer=scryfall&utm_campaign=card_prices&utm_medium=text&utm_source=scryfall","cardhoarder":"https://www.cardhoarder.com/cards/22609?affiliate_id=scryfall&ref=card-profile&utm_campaign=affiliate&utm_medium=card&utm_source=scryfall"}}''')
    filedetails: CardDetails = {
         'file': "C:\\Users\\Sam\\Documents\\FblthpFoundries\\fblthp\\art\\DALL-E\\Sudden Death.png", 'name': card.get('name', ''),
        'set': '', 'artist': '', 'creator': '', 'number': ''
    }
    layout = layout_map.get(card.get('layout', 'normal'))
    layout(card, filedetails)
    layout.template_file = "C:\\Users\\Sam\\Documents\\FblthpFoundries\\fblthp\\Proxyshop\\templates\\borderless-vector.psd"
    layout.art_file = "C:\\Users\\Sam\\Documents\\FblthpFoundries\\fblthp\\art\\DALL-E\\Sudden Death.png"
    bvt = BorderlessVectorTemplate(layout=layout)
    bvt.execute()
    