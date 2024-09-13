from PyQt6.QtWidgets import QListWidgetItem

class Card(QListWidgetItem):
    def __init__(self, cardDict):
        super().__init__()
        self.name = cardDict['name']
        self.oracle = cardDict['oracle_text']
        self.type = cardDict['type_line']
        self.mc = cardDict['mana_cost'] if 'mana_cost' in cardDict else None
        self.flavor = cardDict['flavor_text'] if 'flavor_cost' in cardDict else None
        self.power = cardDict['power'] if 'power' in cardDict else None
        self.toughness = cardDict['toughness'] if 'toughness' in cardDict else None
        self.loyalty = cardDict['loyalty'] if 'loyalty' in cardDict else None

        self.setText(f'{self.name}, {self.type}, {self.mc if not self.mc == 'nan' else ''}:\n {self.oracle}\n{self.power + '/' + self.toughness if self.power else ''}{self.loyalty if self.loyalty else ''}')
