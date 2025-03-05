from requests import get
from json import loads
import pandas as pd
import random

scryfallEndpoint = 'https://api.scryfall.com/' 

def getSetList():
    setTypes = ['core', 'expansion', 'masters', 'draft_innovation']
    allSets = loads(get(f'{scryfallEndpoint}/sets').text)['data']
    draftSets = [] 
    for s in allSets:
        if s['set_type'] in setTypes:
            draftSets.append({'name': s['name'], 'code': s['code']})

    return draftSets


def getSet(set='mh3'):
    setURI = loads(get(f'{scryfallEndpoint}sets/{set}').text)['search_uri']
    page = loads(get(setURI).text)
    set = pd.DataFrame(page['data'])
    while page['has_more'] == True:
        page = loads(get(page['next_page']).text)
        cards = pd.DataFrame(page['data'])
        set = pd.concat([set, cards ], axis=0, ignore_index=True)


    return set 

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
            if '//' in self.name:
                #pick first face for double sided cards
                self.img = frame['card_faces'][0]['image_uris']['normal']
                self.doubleFaced = True
                self.back = frame['card_faces'][1]['image_uris']['normal']
            else:
                self.img = frame['image_uris']['normal']
                self.doubleFaced = False
                self.back = None

        def toJson(self, num):
            return {'id':num, 'name':self.name, 'img':self.img,
                     'doubleFaced':self.doubleFaced, 'back':self.back}

    def __init__(self, set: pd.DataFrame):
        #1-6
        commons = set.loc[set['rarity'] == 'common'].loc[~set['type_line'].str.contains('Basic')].sample(n=6)
        #7
        bonus = pd.concat([set.loc[set['rarity'] == 'common'], set.loc[set['rarity'] == 'bonus']]).sample(n=1)
        #8-10
        uncommons = set.loc[set['rarity'] == 'uncommon'].sample(n=3)
        #12
        rarity = 'mythic' if random.randint(1,8) == 1 else 'rare'
        rare = set.loc[set['rarity'] == rarity].sample(n=1) 
        #11 & 14
        foils = set.sample(n=2)
        #13 Do basics good
        land = set.loc[set['type_line'].str.contains('Basic')].sample(n=1)

        self.cards = [self.Card(x) for _,x in commons.iterrows()]
        self.cards.append(self.Card(bonus.iloc[0]))
        self.cards.extend([self.Card(x) for _,x in uncommons.iterrows()])
        self.cards.append(self.Card(rare.iloc[0]))
        self.cards.extend([self.Card(x) for _,x in foils.iterrows()])
        self.cards.append(self.Card(land.iloc[0]))

    def pick(self, idx):
        card = self.cards[idx]
        del self.cards[idx]
        return card
    
    def len(self):
        return len(self.cards)




    def toJson(self):
        pack = []
        i = 0
        for card in self.cards:
            pack.append(card.toJson(i))
            i+=1
        return {'pack':pack}




if __name__ == '__main__':
    Pack(getSet())