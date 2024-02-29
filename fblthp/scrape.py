import json

cards = json.loads(open('magic-cards.json', 'r', encoding="utf-8").read())

for card in cards:
    string = card['name']
    print(string)

