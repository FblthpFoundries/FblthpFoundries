from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal
from helpers.renderers import ProxyshopRenderer, MSERenderer

class RenderTab(QWidget):
    requestCards = pyqtSignal()

    def __init__(self,  parent=None):
        super(RenderTab, self).__init__(parent)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.render_button = QPushButton('Render')
        self.render_button.clicked.connect(self.request)
        self.layout.addWidget(self.render_button)

        self.rendering_widget = QWidget()
        self.layout.addWidget(self.rendering_widget)
        self.renderer = MSERenderer()

    def request(self):
        self.requestCards.emit()

    def changeRenderer(self, string):
        match string:
            case 'ProxyShop':
                self.renderer = ProxyshopRenderer()
            case 'Magic Set Editor':
                self.renderer = MSERenderer()


    def render(self, cards):
        self.renderer.render_cards(cards)
        """for card in self.card_list_widget.get_cards().values():
            renderer = ProxyshopRenderer()
            new_path = renderer.render_card(card, card.image_path)
            card.render_path = new_path
        """
        print('Rendering complete')
    