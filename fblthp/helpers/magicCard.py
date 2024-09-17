from PyQt6.QtWidgets import QListWidgetItem
import uuid, re

class Card(QListWidgetItem):
    def __init__(self, cardDict):
        super().__init__()
        self.uuid = uuid.uuid4()
        self.name = cardDict['name']
        self.oracle_text = cardDict['oracle_text'].replace('<', '{').replace('>', '}')
        self.type_line = cardDict['type_line']
        self.mana_cost = cardDict['mana_cost'].replace('<', '{').replace('>', '}') if 'mana_cost' in cardDict else None
        self.flavor_text = cardDict['flavor_text'] if 'flavor_text' in cardDict else None
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

    def getXML(self, root):
        cardTag = root.createElement('card')

        name = root.createElement('name')
        name.appendChild(root.createTextNode(self.name))
        cardTag.appendChild(name)

        text = root.createElement('text')
        text.appendChild(root.createTextNode(self.oracle_text))
        cardTag.appendChild(text)

        setTag = root.createElement('set')
        setTag.appendChild(root.createTextNode('FFAI'))
        cardTag.appendChild(setTag)

        row = '1'
        if 'land' in self.type_line.lower():
            row = '0'
        elif 'creature ' in self.type_line.lower():
            row = '2'
        elif 'instant' in self.type_line.lower() or 'sorcery' in self.type_line.lower():
            row = '3'

        tableRow = root.createElement('tablerow')
        tableRow.appendChild(root.createTextNode(row))
        cardTag.appendChild(tableRow)

        return cardTag
    
    def parsePlaneswalker(self,text):
        loyaltyRE = r'[\+\-]*[0-9]+:'
        parsed = ''
        lines = text.split('\n')

        lineCounter = 1

        for line in lines:
            if re.search(loyaltyRE, line):
                split = line.split(':')
                parsed += f'\tloyalty_cost_{lineCounter}: {split[0]}\n'
                parsed += f'\tlevel_{lineCounter}_text: {split[1]}\n'
            else:
                parsed += f'\tlevel_{lineCounter}_text: {line}\n'
            lineCounter+=1

        return parsed


    def genMSE(self):
        types = self.type_line.split("-")
        angry = self.oracle_text.replace('\n', '\n\t\t')
        oracle = self.oracle_text if not '\n' in self.oracle_text else f"\n\t\t{angry}"
        flavor = self.flavor_text

        text = 'card:\n'
        if 'planeswalker' in types[0].lower():
            text += '\tstylesheet: m15-mainframe-planeswalker\n\tstylesheet_version: 2024-01-05\n'
        text += f'\tname: {self.name}\n'

        if not 'planeswalker' in types[0].lower():
            text += f'\trule_text: {oracle.replace('} {', '').replace('{','').replace('}', '')}\n'
        else:
            text += self.parsePlaneswalker(self.oracle_text)
        text += f'\tsuper_type: {types[0]}\n'
        if self.image_path:
            print(self.image_path)
            text += f'\timage: {self.name.replace(' ', '')}\n'
        else:
            text += '\timage: gradient\n'
        if len(types) > 1:
            text += f'\tsub_type: {types[1]}\n'
        if self.power:
            text += f'\tpower: {self.power}\n'
        if self.toughness:
            text += f'\ttoughness: {self.toughness}\n'
        if self.loyalty:
            text += f'\tloyalty: {self.loyalty}\n'
        if self.flavor_text:
            text += f'\tflavor_text: <i-flavor>{flavor}</i-flavor>\n'
        if self.mana_cost:
            text += f'\tcasting_cost: {self.mana_cost.replace("} {", "").replace("{","").replace("}", "")}\n'

        return text[:-1]
    

    