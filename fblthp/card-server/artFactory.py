import requests, urllib
from datetime import datetime
from card import Card
from genMSE import createMSE
import  os, subprocess
from pathlib import Path
import logging
BASE_DIR = Path(__file__).resolve().parent
FBLTHP_OUT = BASE_DIR / "images" / "rendered"


class ArtGen:
    def __init__(self, logger: logging.Logger ):
        self.logger = logger
        if not os.path.exists(FBLTHP_OUT):
            os.makedirs(FBLTHP_OUT)

    def getArt(self, batch: list[Card]):
        pass

    def renderBatch(self, batch: list[Card]) -> list[str]:
        MSE_PATH = BASE_DIR / 'Basic-M15-Magic-Pack'
        # Create the ZIP file path
        zipPath = createMSE(batch, 'tmp', self.logger)

        # Define the rendering script
        renderScript = f'for each c in set.cards do write_image_file(c, file: c.name + ".png")'

        # Open the MSE process directly, passing the render script via stdin
        try:
            renderProcess = subprocess.Popen([MSE_PATH / 'mse', '--cli', zipPath],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True)

            # Send the script content to the stdin of the MSE process
            output, error = renderProcess.communicate(input=renderScript)

            if error:
                print(f"Error during rendering: {error}")
            else:
                print(f"Rendering completed. Output: {output}")

        except FileNotFoundError:
            print(f"File not found. Ensure MSE_PATH is correct: {MSE_PATH / 'mse'}")

        for card in batch:
            fileName = card.name[1:] if card.name[0] == ' ' else card.name
            os.rename(BASE_DIR/f'{fileName}.png', FBLTHP_OUT/f'{fileName + str(card.uuid)}.png')
            card.render_path = FBLTHP_OUT/f'{fileName + str(card.uuid)}.png'

        os.remove(BASE_DIR/'tmp.mse-set')
        for file in os.listdir(BASE_DIR):
            if file.endswith('.png'):
                os.remove(file)
