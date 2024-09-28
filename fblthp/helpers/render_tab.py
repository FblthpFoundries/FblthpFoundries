from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from helpers.renderers import ProxyshopRenderer

class RenderTab(QWidget):
    def __init__(self, card_list_widget, parent=None):
        super(RenderTab, self).__init__(parent)
        self.card_list_widget = card_list_widget

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.render_button = QPushButton('Render')
        self.render_button.clicked.connect(self.render)
        self.layout.addWidget(self.render_button)

        self.rendering_widget = QWidget()
        self.layout.addWidget(self.rendering_widget)

    def render(self):
        for card in self.card_list_widget.get_cards().values():
            renderer = ProxyshopRenderer()
            new_path = renderer.render_card(card, card.image_path)
            card.render_path = new_path
        print('Rendering complete')
    