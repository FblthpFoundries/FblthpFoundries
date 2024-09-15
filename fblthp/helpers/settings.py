from PyQt6.QtWidgets import (
    QWidget,QVBoxLayout,QGroupBox,  QScrollArea,QTextEdit, 
    QFormLayout, QCheckBox, QSpinBox, QComboBox,  QLabel, 
)
from PyQt6.QtCore import  pyqtSignal, QSettings

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