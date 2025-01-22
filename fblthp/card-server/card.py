import uuid, re

class Card:
    def __init__(self, cardDict):
        self.uuid = uuid.uuid4()
        self.name = cardDict['name'] if 'name' in cardDict else None
        self.oracle_text = cardDict['oracle_text'].replace('<', '{').replace('>', '}') if 'oracle_text' in cardDict else None
        self.type_line = cardDict['type_line'] if 'type_line' in cardDict else None
        self.mana_cost = cardDict['mana_cost'].replace('<', '{').replace('>', '}') if 'mana_cost' in cardDict else None
        self.flavor_text = cardDict['flavor_text'] if 'flavor_text' in cardDict else None
        self.power = cardDict['power'] if 'power' in cardDict else None
        self.toughness = cardDict['toughness'] if 'toughness' in cardDict else None
        self.loyalty = cardDict['loyalty'] if 'loyalty' in cardDict else None
        self.rarity = cardDict['rarity'] if 'rarity' in cardDict else None
        self.image_path =  None
        self.cmc = 0
        self.colors = ''

        self.findCMC()

    def findCMC(self):
        pip = r'\{.*?\}'
        num = r'[0-9]+'
        colors = r'[WUBRG]|[WUBRG]/[WUBRGP]'
        if not self.mana_cost:
            return
        for p in re.findall(pip, self.mana_cost):
            if re.search(num, p):
                self.cmc += int(p[1:-1])
            else:
                self.cmc += 1
                for c in re.findall(colors, p):
                    for col in c.split('/'):
                        if not col in self.colors and (not col == 'P'):
                            self.colors += col

    def parsePlaneswalker(self,text):
        loyaltyRE = r'[\+\-]*[0-9]+:'
        parsed = ''
        lines = text.split('\n')

        lineCounter = 1

        for line in lines:
            if re.search(loyaltyRE, line):
                split = line.split(':')
                parsed += f'\tloyalty_cost_{lineCounter}: {split[0]}\n'
                parsed += f'\tlevel_{lineCounter}_text: <margin:130:0:0>{split[1]}</margin:130:0:0>\n'
            else:
                parsed += f'\tlevel_{lineCounter}_text: <margin:130:0:0>{line}</margin:130:0:0>\n'
            lineCounter+=1

        return parsed


    def genMSE(self):
        types = self.type_line.split("-")
        angry = self.oracle_text.replace('\n ', '\n\t\t')
        oracle = f'<kw-0>{self.oracle_text}</kw-0>' if not '\n' in self.oracle_text else f"\n\t\t<kw-0>{angry}</kw-0>"
        flavor = self.flavor_text
        if flavor and '\n' in flavor:
            flavor = flavor.replace('\n ', '\n\t\t')

        text = 'card:\n'
        if 'planeswalker' in types[0].lower():
            text += '\tstylesheet: m15-mainframe-planeswalker\n\tstylesheet_version: 2024-01-05\n'
        text += f'\tname: {self.name}\n'

        if not 'planeswalker' in types[0].lower():
            text += f"\trule_text: {oracle.replace('} {', '').replace('{','').replace('}', '')}\n"
        else:
            text += self.parsePlaneswalker(self.oracle_text)
        text += f"\tsuper_type: {types[0]}\n"
        if self.image_path:
            print(self.image_path)
            text += f"\timage: {self.name.replace(' ', '')}\n"
        else:
            text += '\timage: gradient\n'
        if len(types) > 1:
            text += f"\tsub_type: {types[1]}\n"
        if self.power:
            text += f"\tpower: {self.power}\n"
        if self.toughness:
            text += f"\ttoughness: {self.toughness}\n"
        if self.loyalty:
            text += f"\tloyalty: {self.loyalty}\n"
        if self.flavor_text:
            text += f"\tflavor_text: \n\t\t<i-flavor>{flavor}</i-flavor>\n"
        if self.mana_cost:
            if self.mana_cost == 'nan':
                text += f'\tcasting_cost:\n'
            else:
                text += f"\tcasting_cost: {self.mana_cost.replace('} {', '').replace('{','').replace('}', '')}\n"

        return text[:-1]