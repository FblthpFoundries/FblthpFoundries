from typing import IO
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import pyqtSignal
import re
from zipfile import ZipFile
from pathlib import Path
from .magicCard import Card
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent / "images" / "downloaded"

class FileWidget(QWidget):
    saveSignal = pyqtSignal(str)
    addCardSignal = pyqtSignal(Card)
    cockExport = pyqtSignal(str)

    def __init__(self, parent = None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.saveCards = QPushButton('Save Cards')
        self.saveCards.clicked.connect(self.save)
        self.loadCards = QPushButton('Load Cards')
        self.loadCards.clicked.connect(self.load)
        self.export = QPushButton('Export to Cockatrice')
        self.export.clicked.connect(self.exportCards)

        layout.addWidget(self.saveCards)
        layout.addWidget(self.loadCards)
        layout.addWidget(self.export)

    def exportCards(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setDirectory('')

        if dialog.exec():
            return



    def save(self):
        self.saveSignal.emit('saveTest')

    def load(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter('MSE set (*.mse-set)')
        dialog.setDirectory(BASE_DIR.name)

        if dialog.exec():
            files = dialog.selectedFiles()

            if len(files) < 1:
                return
            
            with ZipFile(files[0], 'r') as z:
                print(z.namelist())
                if not 'set' in z.namelist():
                    return
                def saveImage(img):
                    if not img in z.namelist():
                        return None
                    z.extract(img, IMAGE_DIR)
                    return IMAGE_DIR / img
                cards = self.seperateCards(z.open('set'))
                for card in cards:
                    self.addCardSignal.emit(Card({}, card, saveImage))

    def seperateCards(self, file:IO[bytes]) -> list[str]:
        cardMatch = r'^card:'
        cards = []
        add = False

        card = ''

        for line in file:
            line = line.decode('utf-8')
            if re.search(cardMatch, line):
                add = True
                if not card == '':
                    cards.append(card[1:])
                    card = ''
            elif add:
                card += line

        if not card == '':
            cards.append(card[1:])

        print(cards)
        print(len(cards))

        return cards

            



            