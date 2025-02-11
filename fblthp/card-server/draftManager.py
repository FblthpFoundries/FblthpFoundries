from Pack import getSet, Pack
class DraftManager():

    class Room():
        def __init__(self, servePack):
            self.set = getSet()
            self.players = []
            self.passRight = True
            self.round = 1
            self.servePack = servePack

        def remove(self, player):
            for i in range(len(self.players)):
                if self.players[i].id == player:
                    del self.players[i]
                    return

        def addPlayer(self, p):
            self.players.append(p)
            p.seat = len(self.players) - 1

        def startRound(self,):
            for p in self.players:
                p.packQueue.append(Pack(self.set))

            for p in self.players:
                pack = p.packQueue[0]
                self.servePack(p.id, pack.toJson())

        def onPick(self, playerId, pick):
            player = None
            #find player based on ID
            for p in self.players:
                player = p
                if p.id == playerId:
                    break
            #make pick out of pack
            pack = p.packQueue.pop(0)
            player.cards.append(pack.pick(pick))

            if pack.len() == 0:
                #empty pack, check if time for new round
                for p in self.players:
                    if len(p.packQueue) > 0:
                        return #still picking
                    
                if self.round == 3:
                    return #no more rounds
                self.passRight = not self.passRight
                self.round += 1
                self.startRound()
                return

            #pass pack to next player
            nextSeat = player.seat + 1 if self.passRight else player.seat - 1
            nextSeat %= len(self.players)
            nextPlayer = self.players[nextSeat]

            nextPlayer.packQueue.append(pack)
            if len(nextPlayer.packQueue) == 1:
                self.servePack(nextPlayer.id, nextPlayer.packQueue[0].toJson())

            #return early in only one player room
            if player.id == nextPlayer.id:
                return
            
            #serve next pack in queue if exists
            if len(player.packQueue) >= 1:
                self.servePack(player.id, player.packQueue[0].toJson())


    class Player():
        def __init__(self, id, room):
            self.id = id
            self.room = room
            self.cards = []
            self.packQueue = []
            self.seat = -1




    def __init__(self,startRoom,servePack):
        self.room = self.Room(servePack)
        self.players = {}
        self.startRoom = startRoom
        self.servePack = servePack
        self.pack = Pack(getSet())


    def on_connect(self, player):
        print('connect')
        self.players[player] = self.Player(player, self.room) 
        self.room.addPlayer(self.players[player])
        self.startRoom(player)

    def on_disconnect(self, player):
        print(f'{player} disconnected')
        self.room.remove(player)
        del self.players[player]

    def onPack(self ):
        self.room.startRound()
        

    def on_pick(self, player, card):
        self.room.onPick(player, card)
