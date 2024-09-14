from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QListWidget, QPushButton, 
    QInputDialog, QDialog, QProgressBar, QTabWidget, QVBoxLayout, 
    QFormLayout, QCheckBox, QSpinBox, QComboBox, QFileDialog, QLabel, 
    QGroupBox, QLineEdit, QScrollArea, QSizePolicy, QStackedWidget, 
    QTextEdit, QSlider, QHBoxLayout, 
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QSize
from xml.dom import minidom
import sys, os
from helpers.foundry import ChatGPTCardGenerator, LocalCardGenerator
from helpers.foundry import DALLEImageGenerator, SD3ImageGenerator, GoogleImageGenerator
from helpers.magicCard import Card
from helpers import genMSE

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
        self.card_list_widget = card_list_widget
        self.setLayout(QVBoxLayout())

        # Button to generate images
        self.generate_button = QPushButton('Generate Images')
        self.generate_button.clicked.connect(self.generate_images)
        self.layout().addWidget(self.generate_button)

        # Container for image gallery
        self.image_container = QWidget()
        self.image_layout = QGridLayout()
        self.image_container.setLayout(self.image_layout)

        # Scroll area to hold the image gallery
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_container)
        self.layout().addWidget(self.scroll_area)

    def generate_images(self):
        # Clear the current image gallery
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Get card titles from the list widget
        titles = [self.card_list_widget.item(i).text() for i in range(self.card_list_widget.count())]
        num_images = len(titles)

        # Example image generation
        for i in range(num_images):
            title = titles[i]
            image_label = QLabel()
            title_label = QLabel(self.truncate_text(title, 15))  # Truncate text to fit better
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setFixedHeight(30)  # Fixed height for title
            
            # Placeholder image generation
            image = QImage(300, 300, QImage.Format.Format_RGB888)  # Larger placeholder image
            image.fill(Qt.GlobalColor.blue)  # Fill with blue color as placeholder
            pixmap = QPixmap.fromImage(image)
            image_label.setPixmap(pixmap.scaled(QSize(300, 300), Qt.AspectRatioMode.KeepAspectRatio))
            
            # Create a vertical layout for title and image
            vertical_layout = QVBoxLayout()
            vertical_layout.addWidget(title_label)
            vertical_layout.addWidget(image_label)
            vertical_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for tighter layout
            vertical_widget = QWidget()
            vertical_widget.setLayout(vertical_layout)
            
            # Add to grid layout (2 images wide)
            self.image_layout.addWidget(vertical_widget, i // 2, i % 2)

    def load_image(self, file_path):
        if os.path.exists(file_path):
            image = QImage(file_path)
            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                return pixmap
        return None

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

    def truncate_text(self, text, max_length):
        """ Truncate the text to a maximum length and add '...' if it's too long. """
        if len(text) > max_length:
            return text[:max_length] + '...'
        return text

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
        self.card_list_widget = QWidget()

        # Layouts for tab widgets
        self.setup_gen_widget()


        self.image_gen_widget = ImageGenWidget(self.list)
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
        self.card_generator = LocalCardGenerator()
        self.image_generator = GoogleImageGenerator()
    def setup_gen_widget(self):
        self.card_list_layout = QVBoxLayout(self.card_list_widget)

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


    def disableButtons(self, yes):
        for button in self.buttons:
            button.setDisabled(yes)


    def updateProgress(self, value):
        self.progress.setValue(value)

    def onGenerationFinished(self, cards):
        self.loading.done(0)
        self.disableButtons(False)
        for card in cards:
            self.list.addItem(Card(card))

    def reroll(self):
        curr_row = self.list.currentRow()
        if curr_row >= 0:
            newCard = self.card_generator.reroll()
            oldCard = self.list.takeItem(curr_row)
            del oldCard
            self.list.insertItem(curr_row, Card(newCard))
            self.list.setCurrentRow(curr_row)

    def edit(self):
        pass

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
