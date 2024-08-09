import json
import requests

r = requests.get('https://api.scryfall.com/bulk-data/oracle-cards')

assert(r.status_code == 200)

r.close()

jsonURI = json.loads(r.content)['download_uri']

r = requests.get(jsonURI)

cards = json.loads(r.content)
r.close()

#https://scryfall.com/docs/api/cards Currently only loading creatures

features = ['mana_cost', 'name', 'type_line', 'power', 'toughness', 'oracle_text', 'loyalty', 'flavor_text']

cardNums = len(cards)

num = 0

f = open('cards.csv', 'w', encoding='utf-8')

data = ""
for feature in features:
    data += feature + ','
data = data[:-1]

f.write(data)


for card in cards:
    num  += 1
    if  'Token' in card['type_line'] or 'card_faces' in card:
        continue
    data = '\n'
    for feature in features:
        if feature not in card:
            data += '<empty>,'
            continue
        data += '\"' + card[feature].replace("\"", "").replace('\n', ' <nl> ').replace('}{', '} {') + '\",'
    data = data[:-1]
    f.write(data)

    if num % 1000 == 0:
        print(f'{num}/{cardNums}')

f.close()