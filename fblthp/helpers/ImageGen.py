from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QPushButton, 
    QVBoxLayout, QLabel, QHBoxLayout, QSizePolicy, QScrollArea
)
from PyQt6.QtGui import QImage, QPixmap, QMovie, QFont, QPalette, QColor
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize, QSettings
from helpers.foundry import DALLEImageGenerator, SD3ImageGenerator, PixabayImageGenerator, TestImageGenerator, ReplicateImageGenerator
import uuid
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"



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
        elif self.image_gen_name == "Replicate":
            self.image_generator = ReplicateImageGenerator()
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
        elif generator == "Replicate":
            self.image_generator = ReplicateImageGenerator()
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
            for card in self.cards_uuid_dict.values():
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