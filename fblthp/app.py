from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QTabWidget, 
)
from xml.dom import minidom
from os import listdir, remove
from os.path import isfile, join
import sys
from helpers.settings import SettingsWidget
from helpers.cardList import CardListWidget
from helpers.ImageGen import ImageGenWidget
from helpers.file import FileWidget
from helpers import genMSE
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"




class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.setWindowTitle("Fblthp Foundries Cube Generator")
        self.setGeometry(100, 100, 1600, 1000)

        layout = QGridLayout(self)
        self.setLayout(layout)

        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget, 0, 0)

        # Create and set up widgets for the tabs
        self.card_list_widget = CardListWidget(self)

        self.cards = []


        # Layouts for tab widgets


        self.image_gen_widget = ImageGenWidget(self.card_list_widget)
        self.rendering_widget = QWidget()
        self.settings_widget = SettingsWidget()
        self.file_widget = FileWidget()
        self.file_widget.saveSignal.connect(self.toXML)
        self.file_widget.saveSignal.connect(self.toMSE)
        self.settings_widget.text_gen_option_changed.connect(self.card_list_widget.swap_text_generator)
        self.settings_widget.image_gen_option_changed.connect(self.image_gen_widget.swap_image_generator)
        self.settings_widget.dalle_wide_changed.connect(self.image_gen_widget.update_dalle_wide)
        self.settings_widget.dalle_hd_changed.connect(self.image_gen_widget.update_dalle_hd)
        self.settings_widget.dalle_additional_prompt_changed.connect(self.image_gen_widget.update_dalle_add_text)

        self.tab_widget.addTab(self.card_list_widget, 'Current Cards')
        self.tab_widget.addTab(self.image_gen_widget, 'Image Generation')
        self.tab_widget.addTab(self.rendering_widget, 'Card Rendering')
        self.tab_widget.addTab(self.settings_widget, 'Settings')
        self.tab_widget.addTab(self.file_widget, 'File')
        self.show()

    def createSetXML(self, root):
        setTag = root.createElement('set')

        name = root.createElement('name')
        setTag.appendChild(name)
        text = root.createTextNode('FFAI')
        name.appendChild(text)

        longName = root.createElement('longname')
        setTag.appendChild(longName)
        text = root.createTextNode('Fblthp Foundries AI generated cube')
        longName.appendChild(text)

        settype = root.createElement('settype')
        setTag.appendChild(settype)
        text = root.createTextNode('Custom')
        settype.appendChild(text)

        date = root.createElement('releasedate')
        setTag.appendChild(date)
        text = root.createTextNode('2001-09-11')
        date.appendChild(text)

        return setTag


    def toXML(self, fileName):
        root = minidom.Document()

        xml = root.createElement('cockatrice_carddatabase')
        xml.setAttribute('version', '4')
        root.appendChild(xml)

        sets = root.createElement('sets')
        sets.appendChild(self.createSetXML(root))
        xml.appendChild(sets)

        cards = root.createElement('cards')

        for row in range(self.card_list_widget.count()):
            cards.appendChild(self.card_list_widget.item(row).getXML(root))

        xml.appendChild(cards)

        xml_str = root.toprettyxml(encoding = 'utf-8').decode()

        with open(fileName + '.xml', 'w') as f:
            f.write(xml_str)

    def toMSE(self, fileName):
        cards = [self.card_list_widget.item(i) for i in range(self.card_list_widget.count())]
        genMSE.createMSE(fileName, cards)

    def onClose(self):
        imgPath = Path(__file__).parent / 'images' / 'downloaded'

        toDelete = [f for f in listdir(imgPath) if isfile(join(imgPath, f))]

        for f in toDelete:
            if not f == 'gradient.png':
               remove(imgPath/f) 



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)

    sys.exit(app.exec())
