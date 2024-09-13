from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QListWidget, QPushButton, 
    QInputDialog, QDialog, QProgressBar, QTabWidget, QVBoxLayout, 
    QFormLayout, QCheckBox, QSpinBox, QComboBox, QFileDialog, QLabel, QLineEdit
)
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

class SettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up layout
        layout = QFormLayout(self)
        self.setLayout(layout)

        # General Generation Rules
        self.cards_per_batch = QSpinBox()
        self.cards_per_batch.setValue(20)
        layout.addRow('Cards per Batch:', self.cards_per_batch)

        self.theme_gpt_seeding = QCheckBox()
        self.theme_gpt_seeding.setChecked(True)
        layout.addRow('Theme GPT Seeding:', self.theme_gpt_seeding)

        self.theme_override = QCheckBox()
        self.theme_override.setChecked(False)
        layout.addRow('Theme Override:', self.theme_override)

        self.theme_override_config = QPushButton('Select Theme Config')
        self.theme_override_config.clicked.connect(self.select_theme_config)
        self.theme_override_config_path = QLabel('nonexistent.yml')
        layout.addRow('Theme Override Config:', self.theme_override_config_path)

        self.sanity_check = QCheckBox()
        self.sanity_check.setChecked(True)
        layout.addRow('Sanity Check:', self.sanity_check)

        self.custom_art_generation = QCheckBox()
        self.custom_art_generation.setChecked(True)
        layout.addRow('Custom Art Generation:', self.custom_art_generation)

        self.art_generator = QComboBox()
        self.art_generator.addItems(['DALL-E', 'SD3'])
        layout.addRow('Art Generator:', self.art_generator)

        # Card Schema Rules
        self.generate_flavor_text = QCheckBox()
        self.generate_flavor_text.setChecked(True)
        layout.addRow('Generate Flavor Text:', self.generate_flavor_text)

        self.max_flavor_text_length = QSpinBox()
        self.max_flavor_text_length.setValue(100)
        layout.addRow('Max Flavor Text Length:', self.max_flavor_text_length)

        # Rarity Weights (for simplicity, using QSpinBox)
        self.rarity_weights = {
            'common': QSpinBox(),
            'uncommon': QSpinBox(),
            'rare': QSpinBox(),
            'mythic': QSpinBox()
        }
        for rarity, widget in self.rarity_weights.items():
            widget.setValue(10 if rarity == 'common' else (20 if rarity == 'uncommon' else (50 if rarity == 'rare' else 20)))
            layout.addRow(f'{rarity.capitalize()} Weight:', widget)

        # Mana Schema
        # Simplified for brevity
        self.color_count_distribution = QLineEdit('0.08, 0.74, 0.11, 0.06, 0.005, 0.005')
        layout.addRow('Color Count Distribution:', self.color_count_distribution)

        self.cmc_distribution = QLineEdit('0.004975, 0.188756, 0.251741, 0.205174, 0.141393, 0.093930, 0.031045, 0.021891, 0.019104, 0.014627, 0.009154, 0.006368, 0.004577, 0.003682, 0.002687, 0.000896')
        layout.addRow('CMC Distribution:', self.cmc_distribution)

        self.pip_distribution = QLineEdit('0.89, 0.1, 0.01')
        layout.addRow('Pip Distribution:', self.pip_distribution)

        # Typeline
        self.card_type_weights = {
            'creature': QSpinBox(),
            'instant': QSpinBox(),
            'sorcery': QSpinBox(),
            'enchantment': QSpinBox(),
            'artifact': QSpinBox(),
            'planeswalker': QSpinBox(),
            'land': QSpinBox()
        }
        for card_type, widget in self.card_type_weights.items():
            widget.setValue(40 if card_type == 'creature' else (15 if card_type in ['instant', 'sorcery'] else 10))
            layout.addRow(f'{card_type.capitalize()} Weight:', widget)

        # Artifact Weights
        self.vehicle_weight = QSpinBox()
        self.vehicle_weight.setValue(10)
        layout.addRow('Vehicle Weight:', self.vehicle_weight)

        self.equipment_weight = QSpinBox()
        self.equipment_weight.setValue(10)
        layout.addRow('Equipment Weight:', self.equipment_weight)

        self.other_weight = QSpinBox()
        self.other_weight.setValue(80)
        layout.addRow('Other Weight:', self.other_weight)

        self.override_color_weights = {
            'colorless': QSpinBox(),
            'white': QSpinBox(),
            'blue': QSpinBox(),
            'black': QSpinBox(),
            'red': QSpinBox(),
            'green': QSpinBox(),
            'multicolor': QSpinBox()
        }
        for color, widget in self.override_color_weights.items():
            widget.setValue(58 if color == 'colorless' else (8 if color in ['white', 'blue', 'black', 'red', 'green'] else 2))
            layout.addRow(f'{color.capitalize()} Color Weight:', widget)

        # Ability and Synergy Settings
        self.regenerate_custom_abilities = QCheckBox()
        self.regenerate_custom_abilities.setChecked(True)
        layout.addRow('Regenerate Custom Abilities:', self.regenerate_custom_abilities)

        self.load_abilities = QCheckBox()
        self.load_abilities.setChecked(True)
        layout.addRow('Load Abilities:', self.load_abilities)

        self.ability_file = QPushButton('Select Ability File')
        self.ability_file.clicked.connect(self.select_ability_file)
        self.ability_file_path = QLabel('abilities.json')
        layout.addRow('Ability File:', self.ability_file_path)

        self.enable_synergy = QCheckBox()
        self.enable_synergy.setChecked(False)
        layout.addRow('Enable Synergy:', self.enable_synergy)

    def select_theme_config(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Theme Config", "", "YAML Files (*.yml);;All Files (*)", options=options)
        if file_path:
            self.theme_override_config_path.setText(file_path)

    def select_ability_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Ability File", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_path:
            self.ability_file_path.setText(file_path)

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.setWindowTitle("Fblthp Foundries Cube Generator")
        self.setGeometry(100, 100, 600, 800)

        layout = QGridLayout(self)
        self.setLayout(layout)

        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget, 0, 0)

        # Create and set up widgets for the tabs
        self.card_list_widget = QWidget()
        self.settings_widget = SettingsWidget()

        self.tab_widget.addTab(self.card_list_widget, 'Current Cards')
        self.tab_widget.addTab(self.settings_widget, 'Settings')

        # Layouts for tab widgets
        self.setup_gen_widget()

        self.show()

        self.factory = cardFactory.Factory()
    def setup_gen_widget(self):
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
