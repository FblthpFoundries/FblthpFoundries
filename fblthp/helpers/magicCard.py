from PyQt6.QtWidgets import QListWidgetItem

class Card(QListWidgetItem):
    def __init__(self, cardDict):
        super().__init__()
        self.name = cardDict['name']
        self.oracle_text = cardDict['oracle_text'].replace('<', '{').replace('>', '}')
        self.type_line = cardDict['type_line']
        self.mana_cost = cardDict['mana_cost'].replace('<', '{').replace('>', '}') if 'mana_cost' in cardDict else None
        self.flavor_text = cardDict['flavor_text'] if 'flavor_cost' in cardDict else None
        self.power = cardDict['power'] if 'power' in cardDict else None
        self.toughness = cardDict['toughness'] if 'toughness' in cardDict else None
        self.loyalty = cardDict['loyalty'] if 'loyalty' in cardDict else None
        self.rarity = cardDict['rarity'] if 'rarity' in cardDict else None
        self.theme = cardDict['theme'] if 'theme' in cardDict else None
        self.adjustment = cardDict['adjustment'] if 'adjustment' in cardDict else None
        self.initial_card = cardDict['initial_card'] if 'initial_card' in cardDict else None
        self.chatgpt_prompt = cardDict['chatgpt_prompt'] if 'chatgpt_prompt' in cardDict else None
        self.image_path = cardDict['image_path'] if 'image_path' in cardDict else None


        self.setText(f"{self.name}, {self.type_line}, {self.mana_cost if not self.mana_cost == 'nan' else ''}:\n {self.oracle_text}\n{self.power + '/' + self.toughness if self.power else ''}{self.loyalty if self.loyalty else ''}")

    def set_image_path(self, path):
        self.image_path = path