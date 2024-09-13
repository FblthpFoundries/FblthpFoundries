from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QListWidget, QPushButton, QInputDialog
import sys
import cardFactory

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.setWindowTitle("Cube Gen")
        self.setGeometry(100, 100, 400, 100)

        layout = QGridLayout(self)
        self.setLayout(layout)

        self.list = QListWidget(self)
        layout.addWidget(self.list, 0, 0, 4, 1)

        gen_button = QPushButton('Gen')
        gen_button.clicked.connect(self.gen)

        reroll_button = QPushButton('Reroll')
        reroll_button.clicked.connect(self.reroll)

        edit_button = QPushButton('Edit')
        edit_button.clicked.connect(self.edit)

        layout.addWidget(gen_button, 0, 1)
        layout.addWidget(reroll_button, 1, 1)
        layout.addWidget(edit_button, 2,1)
        self.show()

        self.factory = cardFactory.Factory()

    def gen(self):
        num, ok = QInputDialog.getInt(self, 'Number to Generate', 'Enter number of cards to generate:', 10, 1, 500, 1)
        if ok:
            cards = self.factory.gen_cube(num)
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
