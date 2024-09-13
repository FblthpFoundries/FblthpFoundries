from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QListWidget, QPushButton, QInputDialog, QDialog, QProgressBar
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
        self.setGeometry(100, 100, 400, 100)

        layout = QGridLayout(self)
        self.setLayout(layout)

        self.list = QListWidget(self)
        layout.addWidget(self.list, 0, 0, 4, 1)

        self.gen_button = QPushButton('Gen')
        self.gen_button.clicked.connect(self.gen)

        self.reroll_button = QPushButton('Reroll')
        self.reroll_button.clicked.connect(self.reroll)

        self.edit_button = QPushButton('Edit')
        self.edit_button.clicked.connect(self.edit)

        self.buttons=[self.gen_button, self.reroll_button, self.edit_button]

        layout.addWidget(self.gen_button, 0, 1)
        layout.addWidget(self.reroll_button, 1, 1)
        layout.addWidget(self.edit_button, 2,1)
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
