from constants import PROXYSHOP_PATH

import sys
import os
sys.path.append(PROXYSHOP_PATH)


# Standard Library Imports
from pathlib import Path

# Third Party Imports
import click

# Local Imports
from src import CON, TEMPLATE_DEFAULTS
from src._loader import TemplateDetails
from src.cards import CardDetails
from src.layouts import layout_map
from src.utils.files import load_data_file


def render_target(name: str = None):
    """Render a single card using JSON data."""

    # Find your art image file
    # Todo: Use target selection in Photoshop, support optional filepath argument
    art_path = Path('art', 'DALL-E', name)
    for suf in ['.jpg', '.png', '.webp']:
        art_file = art_path.with_suffix(suf)
        if art_file.is_file():
            break

    # Check if proper art file was found
    if not art_file.is_file():
        raise OSError(f"No art file matching name '{data_file}' was found!")

    # Load card data from a json file and create fake card "details" dict
    # Todo: Make custom card data an optional filepath argument
    data_file = Path(CON.cwd, 'customs', data_file).with_suffix('.json')
    card = load_data_file(data_file)
    file_details: CardDetails = {
        'file': art_file, 'name': card.get('name', ''),
        'set': '', 'artist': 'DALL-E', 'creator': '', 'number': ''
    }

    # Get appropriate layout class and initialize it
    # Todo: Use the appropriate layout class provided
    layout = layout_map.get(card.get('layout', 'normal'))
    layout(card, file_details)

    # Get appropriate template for this layout
    # Todo: Use default, support optional template name argument or custom defined
    template: TemplateDetails = TEMPLATE_DEFAULTS.get(layout.card_class)
    template_class = template['object'].get_template_class(template['class_name'])
    layout.template_file = template['object'].path_psd

    # Load the template class, create an instance, execute it
    template_class(layout).execute()

if __name__ == "__main__":
    os.chdir("..")
    render_target()
