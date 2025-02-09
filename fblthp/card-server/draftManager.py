from Pack import getSet, Pack
class DraftManager():

    class Room():
        def __init__(self):
            self.players = []
            self.passRight = True

    class Player():
        def __init__(self):
            self.packQueue = []




    def __init__(self,startRoom,servePack):
        self.players = {}
        self.startRoom = startRoom
        self.servePack = servePack
        self.pack = Pack(getSet())


    def on_connect(self, player):
        print('connect')
        self.players[player] = []
        self.startRoom(player)

    def on_disconnect(self, player):
        print(f'{player} disconnected')
        del self.players[player]

    def onPack(self ):
        for player in self.players:
            self.servePack(player, self.pack.toJson())

    def on_pick(self, player, card):
        self.players[player].append(card)
