from requests import get
from json import loads
import pandas as pd
import random

def getSet(set='mh3'):
    return pd.DataFrame(loads(get(f'https://api.scryfall.com/cards/search?q=s={set}').text)['data'])

class Pack():
    """
    1-6:common
    7:common/special guest
    8-10:uncommon
    11: wild foil
    12: 7/8 rare 1/8 mythic
    13: basic/common land
    14: wild foil
    """

    class Card():
        def __init__(self, frame):
            self.name = frame['name']
            print(self.name)

        def toJson(self, num):
            return {'id':num, 'name':self.name}

    def __init__(self, set: pd.DataFrame):
        print(set.loc[set['name'] == 'Island']['name'] == 'Island')
        #1-6
        commons = set.loc[set['rarity'] == 'common'].sample(6)
        #7
        bonus = pd.concat([set.loc[set['rarity'] == 'common'], set.loc[set['rarity'] == 'bonus']]).sample(1)
        #12
        rarity = 'mythic' if random.randint(1,8) == 1 else 'rare'
        rare = set.loc[set['rarity'] == rarity].sample(1) 
        #11 & 14
        foils = set.sample(2)
        #13 add basics to this somehow
        land = set.loc[set['type_line'] == 'Land'].loc[set['rarity'] == 'common'].sample(1)

        self.cards = [self.Card(x) for _,x in commons.iterrows()]
        self.cards.append(self.Card(bonus.iloc[0]))
        self.cards.append(self.Card(rare.iloc[0]))
        self.cards.extend([self.Card(x) for _,x in foils.iterrows()])
        self.cards.append(self.Card(land.iloc[0]))




    def toJson(self):
        pack = []
        i = 0
        for card in self.cards:
            print(card)
            pack.append(card.toJson(i))
            i+=1
        return {'pack':pack}




if __name__ == '__main__':
    Pack(getSet())