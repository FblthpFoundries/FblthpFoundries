class DraftManager():
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
            self.servePack(player, {'pack':'pack'})

    def on_pick(self, player, card):
        self.players[player].append(card)
