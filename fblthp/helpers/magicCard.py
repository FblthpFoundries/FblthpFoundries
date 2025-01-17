from PyQt6.QtWidgets import QListWidgetItem
import uuid, re

class Card(QListWidgetItem):
    def __init__(self, cardDict: dict, text:str = None, saveMethod = None):
        super().__init__()
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
        self.theme = cardDict['theme'] if 'theme' in cardDict else None
        self.adjustment = cardDict['adjustment'] if 'adjustment' in cardDict else None
        self.initial_card = cardDict['initial_card'] if 'initial_card' in cardDict else None
        self.chatgpt_prompt = cardDict['chatgpt_prompt'] if 'chatgpt_prompt' in cardDict else None
        self.image_path = cardDict['image_path'] if 'image_path' in cardDict else None
        self.cmc = 0
        self.colors = ''

        if text and saveMethod:
            self.init2ElectricBoogaloo(text, saveMethod)

        self.findCMC()

        self.setText(f"{self.name}, {self.type_line}, {self.mana_cost if not self.mana_cost == 'nan' else ''}:\n {self.oracle_text}\n{self.power + '/' + self.toughness if self.power else ''}{self.loyalty if self.loyalty else ''}")

    def init2ElectricBoogaloo(self, text: str, saveMethod):
        xmlRE = r'(<.*?>|</.*?>)'
        planesCost = r'loyalty_cost_[0-9]+'
        planesText = r'level_[0-9]+_text'
        text = text.replace('\r\n\t\t', '\n ')
        text = re.sub(xmlRE, '', text)
        elements = text.split('\r\n\t')
        superType, subType = None, None

        planeswalkerText = {}

        for element in elements:
            colon = element.find(':')
            if colon == len(element) -1:
                continue
            name, value = element[:colon], element[colon + 1:] 
            if len(value) > 1 and value[0:2] == ' \n':
                value = value[2:]

            if re.search(planesCost, name) or re.search(planesText, name):
                planeswalkerText[name] = value
                continue
            
            match name:
                case 'name':
                    self.name = value
                case 'rule_text':
                    self.oracle_text = value
                case 'image':
                    self.image_path = saveMethod(value[1:])
                case 'super_type':
                    superType = value
                case 'sub_type':
                    subType = value
                case 'toughness':
                    self.toughness = value
                case 'power':
                    self.power = value
                case 'loyalty':
                    self.loyalty = value
                case 'flavor_text':
                    self.flavor_text = value
                case 'casting_cost':
                    self.mana_cost = self.rePip(value)

        self.type_line = superType if not subType else f'{superType} - {subType}'
        
        if 'Planeswalker' in superType:
            oracle = ''
            lineCount = 1
            while len(planeswalkerText) > 0:
                if f'loyalty_cost_{lineCount}' in planeswalkerText:
                    oracle += f"{planeswalkerText[f'loyalty_cost_{lineCount}']}: "
                    del planeswalkerText[f'loyalty_cost_{lineCount}']
                if f'level_{lineCount}_text' in planeswalkerText:
                    oracle += f"{planeswalkerText[f'level_{lineCount}_text']}\n"
                    del planeswalkerText[f'level_{lineCount}_text']
                lineCount += 1

            self.oracle_text = oracle[:-1]

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

    def rePip(self, mc):
        print(mc)
        manaPatterns = [r'([0-9]+)', r'([WUBRG])', r'([WUBRG]/[WUBRGP])']
        for pattern in manaPatterns:
            mc = re.sub(pattern, r'{\1}', mc)
        print(mc)

        return mc

    def set_image_path(self, path):
        self.image_path = path

    def getXML(self, root):
        cardTag = root.createElement('card')

        prop = root.createElement('prop')

        layout = root.createElement('layout')
        layout.appendChild(root.createTextNode('normal'))
        prop.appendChild(layout)

        name = root.createElement('name')
        name.appendChild(root.createTextNode(self.name))
        cardTag.appendChild(name)

        mana = root.createElement('manacost')
        mana.appendChild(root.createTextNode(self.mana_cost.replace('} {', '').replace('{', '').replace('}','')))
        prop.appendChild(mana)
        


        setTag = root.createElement('set')
        setTag.setAttribute('rarity', 'common')
        setTag.appendChild(root.createTextNode('FFAI'))
        cardTag.appendChild(setTag)

        cmcTag = root.createElement('cmc')
        cmcTag.appendChild(root.createTextNode(str(self.cmc)))
        prop.appendChild(cmcTag)

        typeTag = root.createElement('type')
        typeTag.appendChild(root.createTextNode(self.type_line))
        prop.appendChild(typeTag)

        if not self.colors == '':
            colors = root.createElement('colors')
            identity = root.createElement('colorIdentity')
            colors.appendChild(root.createTextNode(self.colors))
            identity.appendChild(root.createTextNode(self.colors))
            prop.appendChild(colors)
            prop.appendChild(identity)

        if self.power and self.toughness:
            pt = root.createElement('pt')
            pt.appendChild(root.createTextNode(f'{self.power}/{self.toughness}'))
            prop.appendChild(pt)
        if self.loyalty:
            loyalty = root.createElement('loyalty')
            loyalty.appendChild(root.createTextNode(self.loyalty))
            prop.appendChild(loyalty)

        legal = root.createElement('format-limited')
        legal.appendChild(root.createTextNode('legal'))
        prop.appendChild(legal)

        row = '1'
        if 'land' in self.type_line.lower():
            row = '0'
        elif 'creature ' in self.type_line.lower():
            row = '2'
        elif 'instant' in self.type_line.lower() or 'sorcery' in self.type_line.lower():
            row = '3'

        tableRow = root.createElement('tablerow')
        tableRow.appendChild(root.createTextNode(row))
        prop.appendChild(tableRow)

        mainType = root.createElement('maintype')

        if row == '0':
            mainType.appendChild(root.createTextNode('Land'))
        elif row == '2':
            mainType.appendChild(root.createTextNode('Creature'))
        elif row == '3':
            mainType.appendChild(root.createTextNode('Instant' if 'instant' in self.type_line.lower() else 'Sorcery'))
        else:
            superTypes = self.type_line.split(' ')
            for bigType in superTypes:
                if (not 'legendary' in bigType.lower()) and (not 'kindred' in bigType.lower()):
                    mainType.appendChild(root.createTextNode(bigType))
                    break

        prop.appendChild(mainType)

        text = root.createElement('text')
        text.appendChild(root.createTextNode(self.oracle_text))
        cardTag.appendChild(text)

        cardTag.appendChild(prop)

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
    

    