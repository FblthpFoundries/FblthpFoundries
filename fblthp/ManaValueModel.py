import requests, json, random

class ManaValue:
    class Distribution:
        def __init__(self):
            self.data = {}
            self.total = 0

        def __str__(self):
            strRep = '{\n'
            for key in self.data:
                strRep += f'    {key}: {self.data[key]}\n'
            strRep += '}'
            return strRep

        def add(self, key):
            if key == '':
                return
            if not key in self.data:
                self.data[key] = 1
            else:
                self.data[key] += 1
            self.total += 1

        def sample(self):
            r = random.randint(0, self.total)
            for key in self.data:
                r -= self.data[key]
                if r <= 0:
                    return key

    def __init__(self):
        self.costMap = {}

        #max cmc is 16
        for i in range(0, 17):
            self.costMap[i] = self.Distribution()
     

        r = requests.get('https://api.scryfall.com/bulk-data/oracle-cards')

        assert(r.status_code == 200)
        r.close()

        jsonURI = json.loads(r.content)['download_uri']

        r = requests.get(jsonURI)

        cards = json.loads(r.content)
        r.close()

        for c in cards:
            if 'Token' in c['type_line'] or 'card_faces' in c:
                continue
            if not 'paper' in c['games']:
                continue
            if not 'Creature' in c['type_line']:
                continue

            bucket = self.costMap[int(c['cmc'])]
            bucket.add(c['mana_cost'])

    def sample(self, cmc: int):
        return self.costMap[cmc].sample()
    
    def printSample(self, cmc: int ):
        print(self.costMap[cmc])

if __name__ == '__main__':
    dist = ManaValue()
    dist.printSample(0)
    for i in range(0, 50):
        print(dist.sample(0))


        