from PyQt6.QtWidgets import (
    QWidget, QPushButton, QHBoxLayout,
)
from PyQt6.QtCore import pyqtSignal

class FileWidget(QWidget):
    saveSignal = pyqtSignal(str)

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
        pass