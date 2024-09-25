from PyQt6.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import pyqtSignal
from zipfile import ZipFile
from pathlib import Path
from .magicCard import Card
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR.parent / "images" / "downloaded"

class FileWidget(QWidget):
    saveSignal = pyqtSignal(str)
    addCardSignal = pyqtSignal(Card)

    def __init__(self, parent = None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.saveCards = QPushButton('Save Cards')
        self.saveCards.clicked.connect(self.save)
        self.loadCards = QPushButton('Load Cards')
        self.loadCards.clicked.connect(self.load)
        self.genArt = QPushButton('Generate Art')

        layout.addWidget(self.saveCards)
        layout.addWidget(self.loadCards)
        layout.addWidget(self.genArt)



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
                if not 'set' in z.namelist():
                    return
                def saveImage(img):
                    z.extract(img, IMAGE_DIR)
                    return IMAGE_DIR / img
                cards = str(z.open('set').read()).split('card:')[1:]
                for card in cards:
                    self.addCardSignal.emit(Card({}, card, saveImage))

            