class DraftManager():

    class Pack():
        class Card():
            def __init__(self, num):
                self.id = num
                self.name = f'card{num}'

            def toJson(self):
                return {'id':self.id, 'name':self.name}
        def __init__(self):
            self.cards = [self.Card(i) for i in range(15)]

        def toJson(self):
            return {'pack':[card.toJson() for card in self.cards]}


    def __init__(self,startRoom,servePack):
        self.players = {}
        self.startRoom = startRoom
        self.servePack = servePack

    def on_connect(self, player):
        print('connect')
        self.players[player] = []
        self.startRoom(player)

    def on_disconnect(self, player):
        print(f'{player} disconnected')
        del self.players[player]

    def pack(self, ):
        for player in self.players:
            self.servePack(player, self.Pack().toJson())

    def on_pick(self, player, card):
        self.players[player].append(card)
