from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QListWidget, QPushButton, 
    QInputDialog, QDialog, QProgressBar, QTabWidget, QVBoxLayout, 
    QFormLayout, QCheckBox, QSpinBox, QComboBox, QFileDialog, QLabel, 
    QGroupBox, QLineEdit, QScrollArea, QSizePolicy, QStackedWidget, 
    QTextEdit, QSlider, QHBoxLayout, 
)
from PyQt6.QtGui import QImage, QPixmap, QMovie, QFont, QPalette, QColor
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize
from xml.dom import minidom
import sys, os
import uuid
from helpers.foundry import ChatGPTCardGenerator, LocalCardGenerator
from helpers.foundry import DALLEImageGenerator, SD3ImageGenerator, PixabayImageGenerator, TestImageGenerator
from helpers.magicCard import Card
from helpers import genMSE
from pathlib import Path
import threading
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
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

class ImageGenerationThread(QThread):
    image_generated = pyqtSignal(uuid.UUID, str)  # uuid, image_path

    def __init__(self, image_generator, card):
        super().__init__()
        self.image_generator = image_generator
        self.card = card

    def run(self):
        try:
            image_path = self.image_generator.generate_image(self.card)
            self.card.image_path = image_path
            self.image_generated.emit(self.card.uuid, image_path)
        except Exception as e:
            print(f"Error generating image for card {self.card.name}: {e}")

class CardImageWidget(QWidget):
    def __init__(self, card, parent=None):
        super().__init__(parent)
        self.card = card
        self.generating = False
        self.failed = False

        # Create layout and labels
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.title_label = QLabel(card.name)

        # Style the title label (centered, clean font, underlined)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(False)  # No need for word wrapping
        self.title_label.setFont(QFont('Arial', 14, QFont.Weight.Medium))  # Start with a default font size and weight
        self.title_label.setStyleSheet("color: #E0E0E0; text-decoration: underline;")  # Underline + light text

        # Initial title size adjustments
        self.title_label.setFixedHeight(40)  # Fixed height
        self.title_label.setFixedWidth(320)  # Give more horizontal room

        # Adjust the font size to fit
        self.adjust_font_size()

        # Style the image label (center the image)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(280, 280)  # A bit more compact

        # Add title and image to the layout
        self.layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.setContentsMargins(10, 10, 10, 10)  # Cleaner margins
        self.layout.setSpacing(5)  # Subtle spacing

        # Add modern dark mode styling with reduced rounded corners and underlined title
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;  /* Dark background */
                border: 1px solid #4c4c4c;  /* Subtle border */
                border-radius: 5px;  /* Much less rounded corners */
                padding: 10px;
            }
            QLabel {
                color: #E0E0E0;  /* Lighter text */
            }
        """)

        # Add drop shadow to the card for depth
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))  # Dark background
        self.setPalette(palette)

        self.uuid = card.uuid
        
        # If an image exists, load it. Otherwise, start the generation process.
        self.update_image()
    def adjust_font_size(self):
        """Automatically adjusts the font size to fit within the title label."""
        font = self.title_label.font()
        font_size = font.pointSize()

        # Reduce font size until the text fits within the width of the label
        while self.title_label.fontMetrics().horizontalAdvance(self.card.name) > self.title_label.width() and font_size > 8:
            font_size -= 1
            font.setPointSize(font_size)
            self.title_label.setFont(font)
    def update_image(self):
        if self.card.image_path:
            image = QImage(self.card.image_path)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(QSize(300, 300), Qt.AspectRatioMode.KeepAspectRatio))
        elif self.generating:
            image = QMovie(str(IMAGE_DIR / "defaults" / "loading.gif"))
            image.setScaledSize(QSize(150, 150))
            self.image_label.setMovie(image)
            self.image_label.resize(QSize(300, 300))
            image.start()
            # image = QImage(str(IMAGE_DIR / "defaults" / "loading.gif"))
            # pixmap = QPixmap.fromImage(image)
            # self.image_label.setPixmap(pixmap.scaled(QSize(300, 300), Qt.AspectRatioMode.KeepAspectRatio))
        elif self.failed:
            image = QImage(str(IMAGE_DIR / "defaults" / "failed.png"))
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(QSize(300, 300), Qt.AspectRatioMode.KeepAspectRatio))
        else:
            image = QImage(str(IMAGE_DIR / "defaults" / "no-image-available.png"))
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(QSize(300, 300), Qt.AspectRatioMode.KeepAspectRatio))
        
class SettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the main layout for the SettingsWidget
        main_layout = QVBoxLayout(self)

        # Create a scrollable area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Create a container widget for the scrollable area
        container_widget = QWidget()
        container_layout = QVBoxLayout(container_widget)
        container_widget.setLayout(container_layout)

        # Add sections to the container widget
        self.add_general_settings(container_layout)
        self.add_card_text_generation_settings(container_layout)
        self.add_image_generation_settings(container_layout)
        self.add_card_rendering_settings(container_layout)  # Added section
        self.update_image_gen_settings(0)
        self.update_text_gen_settings(0)

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(container_widget)

        # Add the scroll area to the main layout
        main_layout.addWidget(scroll_area)

    def add_general_settings(self, layout):
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout()
        general_group.setLayout(general_layout)

        # Add general settings widgets
        self.cards_per_batch = QSpinBox()
        self.cards_per_batch.setValue(20)

        self.save_intermediates = QCheckBox('Save Intermediate Files')

        general_layout.addRow('Cards per Batch:', self.cards_per_batch)
        general_layout.addRow(self.save_intermediates)

        layout.addWidget(general_group)

    def add_card_text_generation_settings(self, layout):
        card_text_group = QGroupBox("Card Text Generation")
        card_text_layout = QFormLayout()
        card_text_group.setLayout(card_text_layout)

        # Option selector
        self.text_gen_option = QComboBox()
        self.text_gen_option.addItems(['Local GPT-2', 'GPT-4o-mini'])
        self.text_gen_option.currentIndexChanged.connect(self.update_text_gen_settings)
        card_text_layout.addRow('Text Generation Option:', self.text_gen_option)

        # GPT-2 specific settings

        self.gpt2_group = []
        for widget in self.gpt2_group:
            widget.setVisible(False)

        # GPT-4o-mini specific settings

        self.theme_seeding = QCheckBox('Theme Seeding')
        self.sanity_check = QCheckBox('Sanity Check')
        
        self.gpt4o_mini_group = [self.theme_seeding, self.sanity_check]
        for widget in self.gpt4o_mini_group:
            widget.setVisible(False)

        # Add settings widgets
        card_text_layout.addRow(self.theme_seeding)
        card_text_layout.addRow(self.sanity_check)
        layout.addWidget(card_text_group)

    def add_image_generation_settings(self, layout):
        image_gen_group = QGroupBox("Image Generation")
        image_gen_layout = QFormLayout()
        image_gen_group.setLayout(image_gen_layout)

        # Option selector
        self.image_gen_option = QComboBox()
        self.image_gen_option.addItems(['SD3', 'DALL-E', 'Internet Search']) 
        self.image_gen_option.currentIndexChanged.connect(self.update_image_gen_settings)
        image_gen_layout.addRow('Image Generation Option:', self.image_gen_option)

        # SD3 specific settings
        self.sd3_iterations_label = QLabel('Iterations:')
        self.sd3_iterations = QSpinBox()
        self.sd3_iterations.setValue(30)
        self.sd3_iterations.setMinimum(0)
        self.sd3_iterations.setMaximum(50)

        self.sd3_setting_group = [self.sd3_iterations, self.sd3_iterations_label]
        for widget in self.sd3_setting_group:
            widget.setVisible(False)

        # DALL-E specific settings
        self.dalle_wide = QComboBox()
        self.dalle_wide.addItems(["1024x1024", "1792x1024"])
        self.dalle_hd = QCheckBox('HD')

        # Additional prompting for DALL-E
        self.dalle_prompt_chars_label = QLabel('Maximum Image Prompt Size (chars):')
        self.dalle_prompt_chars = QSlider(Qt.Orientation.Horizontal, self)
        self.dalle_prompt_chars.setValue(800)
        self.dalle_prompt_chars.setMinimum(400)
        self.dalle_prompt_chars.setMaximum(4000)
        self.dalle_prompt_chars.setTickInterval(400)
        self.dalle_prompt_chars.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.dalle_prompt_chars_value = QLabel('800')
        self.dalle_prompt_chars.valueChanged.connect(lambda value: self.dalle_prompt_chars_value.setText(str(value)))


        self.additional_prompting_label = QLabel('Additional Prompting:')
        self.additional_prompting_setting = QTextEdit()
        self.additional_prompting_setting.setPlaceholderText('Enter additional prompting...')
        
        self.dalle_setting_group = [self.dalle_hd, self.dalle_wide, self.dalle_prompt_chars, self.dalle_prompt_chars_label, self.dalle_prompt_chars_value, self.additional_prompting_label, self.additional_prompting_setting]
        for widget in self.dalle_setting_group:
            widget.setVisible(False)

        # Internet Search specific settings

        self.internet_search_group = []
        for widget in self.internet_search_group:
            widget.setVisible(False)

        

        # Add settings widgets
        image_gen_layout.addRow(self.sd3_iterations_label, self.sd3_iterations)
        image_gen_layout.addRow(self.dalle_wide)
        image_gen_layout.addRow(self.dalle_hd)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.dalle_prompt_chars)
        slider_layout.addWidget(self.dalle_prompt_chars_value)
        slider_widg = QWidget()
        slider_widg.setLayout(slider_layout)

        image_gen_layout.addRow(self.dalle_prompt_chars_label, slider_widg)
        image_gen_layout.addRow(self.additional_prompting_label, self.additional_prompting_setting)  # Added settings
        layout.addWidget(image_gen_group)

    def add_card_rendering_settings(self, layout):
        card_template_group = QGroupBox("Card Rendering")
        card_template_layout = QFormLayout()
        card_template_group.setLayout(card_template_layout)

        # Option selector
        self.card_template_option = QComboBox()
        self.card_template_option.addItems(['Proxyshop', 'HTML Render'])
        card_template_layout.addRow('Card Rendering Option:', self.card_template_option)

        layout.addWidget(card_template_group)

    def update_text_gen_settings(self, index):
        # Show/hide specific settings and labels based on selected option
        for widget in self.gpt2_group:
            widget.setVisible(index == 0)
        for widget in self.gpt4o_mini_group:
            widget.setVisible(index == 1)

    def update_image_gen_settings(self, index):
        # Show/hide specific settings and labels based on selected option
        for widget in self.sd3_setting_group:
            widget.setVisible(index == 0)
        for widget in self.dalle_setting_group:
            widget.setVisible(index == 1)
        for widget in self.internet_search_group:
            widget.setVisible(index == 2)

class ImageGenWidget(QWidget):
    def __init__(self, card_list_widget, parent=None):
        super().__init__(parent)

        self.image_generator = DALLEImageGenerator()     #   TODO: SETTINGS

        self.card_list_widget = card_list_widget
        self.setLayout(QVBoxLayout())
        buttonbar = QWidget()
        buttonbar.setLayout(QHBoxLayout())
        self.load_button = QPushButton('Load Current Cards')
        self.load_button.clicked.connect(self.load_cards)
        buttonbar.layout().addWidget(self.load_button)
        # Button to generate images
        self.generate_button = QPushButton('Generate Images')
        self.generate_button.clicked.connect(self.generate_images)
        buttonbar.layout().addWidget(self.generate_button)
        self.layout().addWidget(buttonbar)

        # Container for image gallery
        self.image_container = QWidget()
        self.image_layout = QGridLayout()
        self.image_container.setLayout(self.image_layout)

        # Scroll area to hold the image gallery
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_container)
        self.layout().addWidget(self.scroll_area)
    def load_cards(self):
        self.cards_uuid_dict = self.card_list_widget.get_cards()
        # Clear the current image gallery
        self._reload_cards()
    
    def generate_images(self):
        if not isinstance(self.image_generator, SD3ImageGenerator):
        # Launch separate threads for each card without images using QThread
            self.threads = []
            for card in self.cards_uuid_dict.values():
                if not card.image_path:
                    thread = ImageGenerationThread(self.image_generator, card)
                    thread.image_generated.connect(self._update_image)
                    self.widgies[card.uuid].failed = False
                    self.widgies[card.uuid].generating = True
                    self.widgies[card.uuid].update_image()
                    thread.start()
                    self.threads.append(thread)
        else:
            # Sequential image generation for SD3ImageGenerator
            for card in self.cards:
                if not card.image_path:
                    try:
                        card.image_path = self.image_generator.generate_image(card)
                    except Exception as e:
                        print(f"Error generating image for card {card}: {e}")
        #self._reload_cards()
    
    def _update_image(self, uuid, img_path):
        self.cards_uuid_dict[uuid].image_path = img_path
        for i in range(self.image_layout.count()):
            widget = self.image_layout.itemAt(i).widget()
            if widget:
                if widget.uuid == uuid:
                    if not img_path:
                        widget.generating = False
                        widget.failed = True
                    widget.update_image()
                    break

    def _reload_cards(self):
        self.widgies = {}
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Get card titles from the list widget

        # Example image generation
        for i, card in enumerate(self.cards_uuid_dict.values()):
            
            widg = CardImageWidget(card)
            self.widgies[card.uuid] = widg
            
            # Add to grid layout (2 images wide)
            self.image_layout.addWidget(widg, i // 2, i % 2)

    def update_gallery(self, image_paths):
        # Clear existing images
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Add new images to the gallery
        titles = [self.card_list_widget.item(i).text() for i in range(len(image_paths))]
        for i, (path, title) in enumerate(zip(image_paths, titles)):
            pixmap = self.load_image(path)
            if pixmap:
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(QSize(300, 300), Qt.AspectRatioMode.KeepAspectRatio))
                title_label = QLabel(self.truncate_text(title, 15))  # Truncate text to fit better
                title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                title_label.setFixedHeight(30)  # Fixed height for title
                
                # Create a vertical layout for title and image
                vertical_layout = QVBoxLayout()
                vertical_layout.addWidget(title_label)
                vertical_layout.addWidget(image_label)
                vertical_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for tighter layout
                vertical_widget = QWidget()
                vertical_widget.setLayout(vertical_layout)
                
                self.image_layout.addWidget(vertical_widget, i // 2, i % 2)

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

class CardListWidget(QWidget):
    def __init__(self, supreme_ruler=None):
        super().__init__()
        self.supreme_ruler = supreme_ruler
        self.card_generator = ChatGPTCardGenerator() #   TODO: SETTINGS
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
        button_layout.addWidget(self.gen_button)
        button_layout.addWidget(self.reroll_button)
        button_layout.addWidget(self.edit_button)

        self.card_list_layout.addLayout(button_layout)
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


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.setWindowTitle("Fblthp Foundries Cube Generator")
        self.setGeometry(100, 100, 800, 600)

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

    def cardXML(self, card, root):
        cardTag = root.createElement('card')

        name = root.createElement('name')
        name.appendChild(root.createTextNode(card.name))
        cardTag.appendChild(name)

        text = root.createElement('text')
        text.appendChild(root.createTextNode(card.oracle))
        cardTag.appendChild(text)

        setTag = root.createElement('set')
        setTag.appendChild(root.createTextNode('FFAI'))
        cardTag.appendChild(setTag)

        row = '1'
        if 'land' in card.type.lower():
            row = '0'
        elif 'creature ' in card.type.lower():
            row = '2'
        elif 'instant' in card.type.lower() or 'sorcery' in card.type.lower():
            row = '3'

        tableRow = root.createElement('tablerow')
        tableRow.appendChild(root.createTextNode(row))
        cardTag.appendChild(tableRow)

        return cardTag

    def toXML(self, fileName):
        root = minidom.Document()

        xml = root.createElement('cockatrice_cardbase')
        xml.setAttribute('version', '4')
        root.appendChild(xml)

        sets = root.createElement('sets')
        sets.appendChild(self.createSetXML(root))
        xml.appendChild(sets)

        cards = root.createElement('cards')

        for row in range(self.list.count()):
            cards.appendChild(self.cardXML(self.list.item(row), root))

        xml.appendChild(cards)

        xml_str = root.toprettyxml(encoding = 'utf-8').decode()

        with open(fileName + '.xml', 'w') as f:
            f.write(xml_str)

    def toMSE(self, fileName):
        cards = [self.list.item(i) for i in range(self.list.count())]
        genMSE.createMSE(fileName, cards)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec())
