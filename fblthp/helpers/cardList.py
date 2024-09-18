from PyQt6.QtWidgets import (
    QWidget,  QListWidget, QPushButton, QLabel,
    QInputDialog, QDialog, QProgressBar, QVBoxLayout, 
)
from PyQt6.QtCore import QThread, pyqtSignal, QSettings
from helpers.magicCard import Card
from helpers.foundry import ChatGPTCardGenerator, LocalCardGenerator

class WorkerThread(QThread):
    progressUpdated = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, card_generator, num, parent=None):
        super().__init__(parent)
        self.card_generator = card_generator
        self.num = num

    def run(self):
        def updateProgress(num):
            self.progressUpdated.emit(int(num * 100))

        cards = self.card_generator.create_cards(self.num, updateProgress)
        self.finished.emit(cards)

class CardListWidget(QWidget):
    def __init__(self, supreme_ruler=None):
        super().__init__()
        self.supreme_ruler = supreme_ruler
        settings = QSettings("FblthpFoundries", "CardGenerator")
        self.card_gen_name = settings.value("text_gen/option", "Local GPT-2")
        if self.card_gen_name == "Local GPT-2":
            self.card_generator = LocalCardGenerator()
        elif self.card_gen_name == "GPT-4o-mini":
            self.card_generator = ChatGPTCardGenerator()

        self.card_list_layout = QVBoxLayout(self)

        self.list = QListWidget(self)

        self.uuid_dict = {}
        self.card_list_layout.addWidget(self.list)

        self.gen_button = QPushButton('Gen')
        self.gen_button.clicked.connect(self.gen)

        self.reroll_button = QPushButton('Reroll')
        self.reroll_button.clicked.connect(self.reroll)

        self.edit_button = QPushButton('Edit')
        self.edit_button.clicked.connect(self.edit)

        self.buttons=[self.gen_button, self.reroll_button, self.edit_button]

        button_layout = QVBoxLayout()
        self.gen_label = QLabel("Generator: " + self.card_gen_name)
        button_layout.addWidget(self.gen_label)
        button_layout.addWidget(self.gen_button)
        button_layout.addWidget(self.reroll_button)
        button_layout.addWidget(self.edit_button)

        self.card_list_layout.addLayout(button_layout)

    def swap_text_generator(self, generator):
        print(generator)
        if generator == "Local GPT-2":
            self.card_generator = LocalCardGenerator()
            self.card_gen_name = "Local GPT-2"
            self.gen_label.setText("Generator: Local GPT-2")
        elif generator == "GPT-4o-mini":
            self.card_generator = ChatGPTCardGenerator()
            self.card_gen_name = "GPT-4o-mini"
            self.gen_label.setText("Generator: GPT-4o-mini")
    def gen(self):
        self.disableButtons(True)
        num, ok = QInputDialog.getInt(self, 'Number to Generate', 'Enter number of cards to generate:', 10, 1, 500, 1)

        if ok:
            self.loading = QDialog(self)
            self.loading.setWindowTitle('Generating Cards...')
            self.loading.resize(320, 130)
            self.progress = QProgressBar(self.loading)
            self.progress.setGeometry(50, 50, 250, 30)

            self.worker = WorkerThread(self.card_generator, num)
            self.worker.progressUpdated.connect(self.updateProgress)
            self.worker.finished.connect(self.onGenerationFinished)
            self.loading.show()
            self.updateProgress(0)
            self.worker.start()

        else:
            self.disableButtons(False)
    def load_cube_settings(self, file):
        if file:
            self.card_generator.load_cube_settings(file)
    def updateProgress(self, value):
        self.progress.setValue(value)
    def onGenerationFinished(self, cards):
        self.loading.done(0)
        self.disableButtons(False)
        for card in cards:
            c = Card(card)
            self.uuid_dict[c.uuid] = c
            self.list.addItem(c)
    def get_cards(self):
        return self.uuid_dict

        

    def reroll(self):
        curr_row = self.list.currentRow()
        if curr_row >= 0:
            newCard = self.card_generator.reroll()
            oldCard = self.list.takeItem(curr_row)
            del self.uuid_dict[oldCard.uuid]
            del oldCard
            c = Card(newCard)
            self.uuid_dict[c.uuid] = c
            self.list.insertItem(curr_row, c)
            self.list.setCurrentRow(curr_row)
    def edit(self):
        pass

    def disableButtons(self, yes):
        for button in self.buttons:
            button.setDisabled(yes)

    def count(self):
        return self.list.count()
    
    def item(self, i):
        return self.list.item(i)
