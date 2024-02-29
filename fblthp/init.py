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

features = ['mana_cost', 'name', 'type_line', 'power', 'toughness', 'oracle_text', 'flavor_text']

data = ""
for feature in features:
    data += feature + ','
data = data[:-1]

for card in cards:
    if 'Creature' not in card['type_line'] or 'Token' in card['type_line'] or 'card_faces' in card:
        continue
    data += '\n'
    for feature in features:
        if feature not in card:
            data += '<empty>,'
            continue
        data += '\"' + card[feature].replace("\"", "").replace('\n', '<nl>') + '\",'
    data = data[:-1]

f = open('cards.csv', 'w', encoding='utf-8')
f.write(data)
f.close()