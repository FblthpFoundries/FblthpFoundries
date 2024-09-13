from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QListWidget, QPushButton, QInputDialog, QDialog, QProgressBar, QTabWidget, QVBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import cardFactory

class WorkerThread(QThread):
    progressUpdated = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, factory, num, parent=None):
        super().__init__(parent)
        self.factory = factory
        self.num = num

    def run(self):
        def updateProgress(num):
            self.progressUpdated.emit(int(num * 100))

        cards = self.factory.gen_cube(self.num, updateProgress)
        self.finished.emit(cards)

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.setWindowTitle("Cube Gen")
        self.setGeometry(100, 100, 600, 400)

        layout = QGridLayout(self)
        self.setLayout(layout)

        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget, 0, 0)

        # Create and set up widgets for the tabs
        self.card_list_widget = QWidget()
        self.settings_widget = QWidget()

        self.tab_widget.addTab(self.card_list_widget, 'Current Cards')
        self.tab_widget.addTab(self.settings_widget, 'Settings')

        # Layouts for tab widgets
        self.card_list_layout = QVBoxLayout(self.card_list_widget)
        self.settings_layout = QVBoxLayout(self.settings_widget)

        self.list = QListWidget(self.card_list_widget)
        self.card_list_layout.addWidget(self.list)

        self.gen_button = QPushButton('Gen')
        self.gen_button.clicked.connect(self.gen)

        self.reroll_button = QPushButton('Reroll')
        self.reroll_button.clicked.connect(self.reroll)

        self.edit_button = QPushButton('Edit')
        self.edit_button.clicked.connect(self.edit)

        self.buttons=[self.gen_button, self.reroll_button, self.edit_button]

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.gen_button)
        button_layout.addWidget(self.reroll_button)
        button_layout.addWidget(self.edit_button)

        self.card_list_layout.addLayout(button_layout)

        self.show()

        self.factory = cardFactory.Factory()

    def gen(self):
        self.disableButtons(True)
        num, ok = QInputDialog.getInt(self, 'Number to Generate', 'Enter number of cards to generate:', 10, 1, 500, 1)

        if ok:
            self.loading = QDialog(self)
            self.loading.setWindowTitle('Generating')
            self.loading.resize(320, 130)
            self.progress = QProgressBar(self.loading)
            self.progress.setGeometry(50, 50, 250, 30)

            self.worker = WorkerThread(self.factory, num)
            self.worker.progressUpdated.connect(self.updateProgress)
            self.worker.finished.connect(self.onGenerationFinished)
            self.loading.show()
            self.worker.start()

        else:
            self.disableButtons(False)


    def disableButtons(self, yes):
        for button in self.buttons:
            button.setDisabled(yes)


    def updateProgress(self, value):
        self.progress.setValue(value)

    def onGenerationFinished(self, cards):
        self.loading.done(0)
        self.disableButtons(False)
        for card in cards:
            self.list.addItem(card)

    def reroll(self):
        curr_row = self.list.currentRow()
        if curr_row >= 0:
            newCard = self.factory.reroll()
            oldCard = self.list.takeItem(curr_row)
            del oldCard
            self.list.insertItem(curr_row, newCard)

    def edit(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec())
