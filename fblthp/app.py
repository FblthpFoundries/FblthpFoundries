from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QListWidget, QPushButton, 
    QInputDialog, QDialog, QProgressBar, QTabWidget, QVBoxLayout, 
    QFormLayout, QCheckBox, QSpinBox, QComboBox, QFileDialog, QLabel, 
    QGroupBox, QLineEdit, QScrollArea, QSizePolicy, QStackedWidget, 
    QTextEdit, QSlider, QHBoxLayout, 
)
from PyQt6.QtGui import QImage, QPixmap, QMovie, QFont, QPalette, QColor
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize, QSettings
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
    regenerate_requested = pyqtSignal(uuid.UUID)
    def __init__(self, card, parent=None):
        super().__init__(parent)
        self.card = card
        self.uuid = card.uuid  # Keep the UUID
        self.generating = False
        self.failed = False

        # Main vertical layout
        self.layout = QVBoxLayout(self)

        # Create Title Layout (Name + Mana Cost)
        title_layout = QHBoxLayout()
        self.title_label = QLabel(self.card.name)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.title_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #E0E0E0; text-decoration: underline;")

        # Mana Cost label (if exists)
        self.mana_cost_label = QLabel(self.card.mana_cost if self.card.mana_cost else "")
        self.mana_cost_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.mana_cost_label.setFont(QFont('Arial', 12, QFont.Weight.Normal))
        self.mana_cost_label.setStyleSheet("color: #E0E0E0;")

        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.mana_cost_label)

        # Image Label (Center the image)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(150, 150)  # Ensure the image doesn't get too tiny
        self.image_label.setMaximumSize(1000, 800)  # Max size for the image

        # Type Line
        self.type_label = QLabel(self.card.type_line)
        self.type_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.type_label.setStyleSheet("color: #E0E0E0;")

        # Oracle Text (Rules text)
        self.oracle_text_label = QLabel(self.card.oracle_text)
        self.oracle_text_label.setFont(QFont('Arial', 11))
        self.oracle_text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oracle_text_label.setWordWrap(True)
        self.oracle_text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.oracle_text_label.setStyleSheet("color: #E0E0E0;")

        # Power / Toughness or Loyalty (if exists)
        self.stats_label = QLabel(self._get_stats_text())
        self.stats_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.stats_label.setStyleSheet("color: #E0E0E0;")

        # Add a button to regenerate art
        self.regenerate_button = QPushButton("Regenerate Art")
        self.regenerate_button.clicked.connect(self.request_regeneration)  # Connect to the regeneration method

        # Make the button red
        self.regenerate_button.setStyleSheet("""
            QPushButton {
                background-color: maroon;
                color: white;
                border: 1px solid #555;
                padding: 5px;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
        """)


        # Add everything to the main layout
        self.layout.addLayout(title_layout)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.type_label)
        self.layout.addWidget(self.oracle_text_label)
        self.layout.addWidget(self.stats_label)
        self.layout.addWidget(self.regenerate_button)

        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)

        # Dark mode styling with reduced rounded corners and a modern look
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border: 1px solid #4c4c4c;
                border-radius: 5px;
                padding: 10px;
            }
            QLabel {
                color: #E0E0E0;
            }
        """)

        # Set the background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
        self.setPalette(palette)

        # Load or generate image
        self.update_image()

    def _get_stats_text(self):
        """Returns the formatted Power/Toughness or Loyalty text based on the card."""
        if self.card.power and self.card.toughness:
            return f"{self.card.power}/{self.card.toughness}"
        elif self.card.loyalty:
            return f"Loyalty: {self.card.loyalty}"
        return ""

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
        """Handles setting the image for the card, including loading/fail states."""
        if self.card.image_path:
            image = QImage(self.card.image_path)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        elif self.generating:
            movie = QMovie(str(IMAGE_DIR / "defaults" / "loading.gif"))
            movie.setScaledSize(QSize(150, 150))
            self.image_label.setMovie(movie)
            movie.start()
        elif self.failed:
            image = QImage(str(IMAGE_DIR / "defaults" / "failed.png"))
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        else:
            image = QImage(str(IMAGE_DIR / "defaults" / "no-image-available.png"))
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def resizeEvent(self, event):
        """Ensure that the image resizes with the window."""
        self.update_image()  # Refresh image scaling when the window is resized
        super().resizeEvent(event)

    def request_regeneration(self):
        self.regenerate_requested.emit(self.uuid)
        self.card.image_path = None
        self.generating = True
        self.regenerate_button.setEnabled(False)
        self.update_image()
        

class SettingsWidget(QWidget):
    text_gen_option_changed = pyqtSignal(str)
    image_gen_option_changed = pyqtSignal(str)
    dalle_hd_changed = pyqtSignal(bool)
    dalle_wide_changed = pyqtSignal(str)
    dalle_additional_prompt_changed = pyqtSignal(str)

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

        # Load saved settings
        self.load_settings()

        self.update_image_gen_settings(0)
        self.update_text_gen_settings(0)

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(container_widget)

        # Add the scroll area to the main layout
        main_layout.addWidget(scroll_area)
        self.update_text_gen_settings(self.text_gen_option.currentIndex())
        self.update_image_gen_settings(self.image_gen_option.currentIndex())

    def add_general_settings(self, layout):
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout()
        general_group.setLayout(general_layout)

        # Add general settings widgets
        self.cards_per_batch = QSpinBox()
        self.cards_per_batch.setValue(20)
        self.cards_per_batch.valueChanged.connect(lambda: self.save_setting("general/cards_per_batch", self.cards_per_batch.value()))

        self.save_intermediates = QCheckBox('Save Intermediate Files')
        self.save_intermediates.toggled.connect(lambda: self.save_setting("general/save_intermediates", self.save_intermediates.isChecked()))

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
        self.text_gen_option.currentTextChanged.connect(self.text_gen_option_changed.emit)
        self.text_gen_option.currentTextChanged.connect(lambda text: self.save_setting("text_gen/option", text))

        card_text_layout.addRow('Text Generation Option:', self.text_gen_option)

        # GPT-2 specific settings (Currently no additional settings in your original code)
        self.gpt2_group = []  # Keep this as it is in case you need to add settings here later.

        # GPT-4o-mini specific settings
        self.theme_seeding = QCheckBox('Theme Seeding')
        self.sanity_check = QCheckBox('Sanity Check')

        self.gpt4o_mini_group = [self.theme_seeding, self.sanity_check]
        for widget in self.gpt4o_mini_group:
            widget.setVisible(False)  # Hidden by default, shown based on selection

        # Add settings widgets for GPT-4o-mini
        card_text_layout.addRow(self.theme_seeding)
        card_text_layout.addRow(self.sanity_check)
        layout.addWidget(card_text_group)

    def add_image_generation_settings(self, layout):
        image_gen_group = QGroupBox("Image Generation")
        image_gen_layout = QFormLayout()
        image_gen_group.setLayout(image_gen_layout)

        # Option selector
        self.image_gen_option = QComboBox()
        self.image_gen_option.addItems(['SD3', 'DALL-E', 'Pixabay', 'Test'])
        self.image_gen_option.currentIndexChanged.connect(self.update_image_gen_settings)
        self.image_gen_option.currentTextChanged.connect(self.image_gen_option_changed.emit)
        self.image_gen_option.currentTextChanged.connect(lambda text: self.save_setting("image_gen/option", text))

        image_gen_layout.addRow('Image Generation Option:', self.image_gen_option)

        # SD3 specific settings
        self.sd3_iterations_label = QLabel('Iterations:')
        self.sd3_iterations = QSpinBox()
        self.sd3_iterations.setValue(30)
        self.sd3_iterations.setMinimum(0)
        self.sd3_iterations.setMaximum(50)
        self.sd3_iterations.valueChanged.connect(lambda: self.save_setting("sd3/iterations", self.sd3_iterations.value()))

        self.sd3_setting_group = [self.sd3_iterations, self.sd3_iterations_label]
        for widget in self.sd3_setting_group:
            widget.setVisible(False)

        # DALL-E specific settings
        self.dalle_wide = QComboBox()
        self.dalle_wide.addItems(["1024x1024", "1792x1024"])
        self.dalle_wide.currentTextChanged.connect(lambda text: self.save_setting("dalle/wide", text))
        self.dalle_wide.currentTextChanged.connect(self.dalle_wide_changed.emit)
        self.dalle_hd = QCheckBox('HD')
        self.dalle_hd.toggled.connect(lambda: self.save_setting("dalle/hd", self.dalle_hd.isChecked()))
        self.dalle_hd.toggled.connect(self.dalle_hd_changed.emit)

        # Additional prompting for DALL-E

        self.additional_prompting_label = QLabel('Additional Prompting:')
        self.additional_prompting_setting = QTextEdit()
        self.additional_prompting_setting.setPlaceholderText('Enter additional prompting...')
        self.additional_prompting_setting.textChanged.connect(lambda: self.save_setting("dalle/additional_prompting", self.additional_prompting_setting.toPlainText()))
        self.additional_prompting_setting.textChanged.connect(self.on_additional_prompt_changed)

        self.dalle_setting_group = [self.dalle_hd, self.dalle_wide, self.additional_prompting_label, self.additional_prompting_setting]
        for widget in self.dalle_setting_group:
            widget.setVisible(False)

        # Add all the settings widgets
        image_gen_layout.addRow(self.sd3_iterations_label, self.sd3_iterations)
        image_gen_layout.addRow(self.dalle_wide)
        image_gen_layout.addRow(self.dalle_hd)


        image_gen_layout.addRow(self.additional_prompting_label, self.additional_prompting_setting)
        layout.addWidget(image_gen_group)
    def on_additional_prompt_changed(self):
        # Get the current text from the QTextEdit
        text = self.additional_prompting_setting.toPlainText()

        # Emit the signal with the text as an argument
        self.dalle_additional_prompt_changed.emit(text)
    def add_card_rendering_settings(self, layout):
        card_template_group = QGroupBox("Card Rendering")
        card_template_layout = QFormLayout()
        card_template_group.setLayout(card_template_layout)

        # Option selector
        self.card_template_option = QComboBox()
        self.card_template_option.addItems(['Proxyshop', 'HTML Render'])
        self.card_template_option.currentTextChanged.connect(lambda text: self.save_setting("card_render/option", text))

        card_template_layout.addRow('Card Rendering Option:', self.card_template_option)

        layout.addWidget(card_template_group)

    def save_setting(self, key, value):
        """Save a setting using QSettings."""
        settings = QSettings("FblthpFoundries", "CardGenerator")
        settings.setValue(key, value)

    def load_settings(self):
        """Load settings using QSettings and apply them to widgets."""
        settings = QSettings("FblthpFoundries", "CardGenerator")

        # Load general settings
        self.cards_per_batch.setValue(int(settings.value("general/cards_per_batch", 20)))
        self.save_intermediates.setChecked(settings.value("general/save_intermediates", False, type=bool))

        # Load text generation settings
        text_gen_option = settings.value("text_gen/option", "Local GPT-2")
        index = self.text_gen_option.findText(text_gen_option)
        if index >= 0:
            self.text_gen_option.setCurrentIndex(index)

        # Load image generation settings
        image_gen_option = settings.value("image_gen/option", "SD3")
        index = self.image_gen_option.findText(image_gen_option)
        if index >= 0:
            self.image_gen_option.setCurrentIndex(index)

        # Load SD3 and DALL-E specific settings
        self.sd3_iterations.setValue(int(settings.value("sd3/iterations", 30)))
        self.dalle_hd.setChecked(settings.value("dalle/hd", False, type=bool))
        self.additional_prompting_setting.setPlainText(settings.value("dalle/additional_prompting", "", type=str))

        # Load card rendering settings
        card_template_option = settings.value("card_render/option", "Proxyshop")
        index = self.card_template_option.findText(card_template_option)
        if index >= 0:
            self.card_template_option.setCurrentIndex(index)

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


class ImageGenWidget(QWidget):
    def __init__(self, card_list_widget, parent=None):
        super().__init__(parent)
        self.threads = []
        settings = QSettings("FblthpFoundries", "CardGenerator")
        self.image_gen_name = settings.value("image_gen/option", "SD3")
        if self.image_gen_name == "Pixabay":
            self.image_generator = PixabayImageGenerator()
        elif self.image_gen_name == "DALL-E":
            self.image_generator = DALLEImageGenerator()
        elif self.image_gen_name == "SD3":
            self.image_generator = SD3ImageGenerator()
        elif self.image_gen_name == "Test":
            self.image_generator = TestImageGenerator()

        self.card_list_widget = card_list_widget



        self.setLayout(QVBoxLayout())
        buttonbar = QWidget()
        buttonbar.setLayout(QHBoxLayout())
        self.generator_name_label = QLabel("Generator: " + self.image_gen_name)
        buttonbar.layout().addWidget(self.generator_name_label)

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

    def swap_image_generator(self, generator):
        if generator == "SD3":
            self.image_generator = SD3ImageGenerator()
        elif generator == "DALL-E":
            self.image_generator = DALLEImageGenerator()
        elif generator == "Pixabay":
            self.image_generator = PixabayImageGenerator()
        elif generator == "Test":
            self.image_generator = TestImageGenerator()
        else:
            raise Exception("Unrecognized Image Generator")
        self.image_gen_name = generator
        self.generator_name_label.setText(f"Generator: {self.image_gen_name}")

    def update_dalle_wide(self, wide):
        self.image_generator.size = wide
    def update_dalle_hd(self, hd):
        self.image_generator.quality = "hd" if hd else "standard"
    def update_dalle_add_text(self, text):
        self.image_generator.additional_prompt = text
    def load_cards(self):
        self.cards_uuid_dict = self.card_list_widget.get_cards()
        # Clear the current image gallery
        self._reload_cards()
    
    def generate_images(self):
        if not isinstance(self.image_generator, SD3ImageGenerator):
        # Launch separate threads for each card without images using QThread
            
            for card in self.cards_uuid_dict.values():
                if not card.image_path:
                    thread = ImageGenerationThread(self.image_generator, card)
                    thread.image_generated.connect(self._update_image)
                    self.widgies[card.uuid].failed = False
                    self.widgies[card.uuid].generating = True
                    self.widgies[card.uuid].regenerate_button.setEnabled(False)
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
                    widget.regenerate_button.setEnabled(True)
                    widget.update_image()
                    break
    
    def regenerate_art_for_card(self, card_uuid):
        """Regenerate art for a specific card by its UUID."""
        card = self.cards_uuid_dict.get(card_uuid)
        if card:
            if not isinstance(self.image_generator, SD3ImageGenerator):
                # Launch a separate thread for this card using QThread
                thread = ImageGenerationThread(self.image_generator, card)
                thread.image_generated.connect(self._update_image)
                self.widgies[card.uuid].failed = False
                self.widgies[card.uuid].generating = True
                self.widgies[card.uuid].update_image()
                thread.start()
                self.threads.append(thread)
            else:
                # Sequential image generation for SD3ImageGenerator
                try:
                    card.image_path = self.image_generator.generate_image(card)
                    self._update_image(card.uuid, card.image_path)
                except Exception as e:
                    print(f"Error generating image for card {card}: {e}")

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

            widg.regenerate_requested.connect(self.regenerate_art_for_card)
            
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


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.setWindowTitle("Fblthp Foundries Cube Generator")
        self.setGeometry(100, 100, 1600, 1000)

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
        self.settings_widget.text_gen_option_changed.connect(self.card_list_widget.swap_text_generator)
        self.settings_widget.image_gen_option_changed.connect(self.image_gen_widget.swap_image_generator)
        self.settings_widget.dalle_wide_changed.connect(self.image_gen_widget.update_dalle_wide)
        self.settings_widget.dalle_hd_changed.connect(self.image_gen_widget.update_dalle_hd)
        self.settings_widget.dalle_additional_prompt_changed.connect(self.image_gen_widget.update_dalle_add_text)

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
        text.appendChild(root.createTextNode(card.oracle_text))
        cardTag.appendChild(text)

        setTag = root.createElement('set')
        setTag.appendChild(root.createTextNode('FFAI'))
        cardTag.appendChild(setTag)

        row = '1'
        if 'land' in card.type_line.lower():
            row = '0'
        elif 'creature ' in card.type_line.lower():
            row = '2'
        elif 'instant' in card.type_line.lower() or 'sorcery' in card.type_line.lower():
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

        for row in range(self.card_list_widget.count()):
            cards.appendChild(self.cardXML(self.card_list_widget.item(row), root))

        xml.appendChild(cards)

        xml_str = root.toprettyxml(encoding = 'utf-8').decode()

        with open(fileName + '.xml', 'w') as f:
            f.write(xml_str)

    def toMSE(self, fileName):
        cards = [self.card_list_widget.item(i) for i in range(self.card_list_widget.count())]
        genMSE.createMSE(fileName, cards)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec())
