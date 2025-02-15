from Pack import getSet, Pack
import uuid, random, threading
class DraftManager():
    class Room():

        MIN_PLAYERS = 4
        def __init__(self, servePack):
            self.set = getSet()
            self.players = []
            self.passRight = True #direction to pass packs
            self.round = 1
            self.servePack = servePack #call back to serve packs via sockets
            self.lastPick = 0 #number of players who have completeted last pick
            self.doneCountLock = threading.Lock()

        def playerFinished(self):
            self.doneCountLock.acquire()
            self.lastPick += 1
            self.doneCountLock.release()

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

        def serve(self, player, pack):
            if player.isRobot:
                roboPick = threading.Thread
            else:
                self.servePack(player, pack)

        def onPick(self, playerId, pick):
            player = None
            #find player based on ID
            for p in self.players:
                player = p
                if p.id == playerId:
                    break
            #make pick out of pack
            nextSeat = player.seat + 1 if self.passRight else player.seat - 1
            nextSeat %= len(self.players)
            nextPlayer = self.players[nextSeat]
        
            player.send(pick, nextPlayer)

            self.doneCountLock.acquire()

            if self.lastPick == len(self.players):
                if self.round == 3:
                    return #no more rounds
                self.passRight = not self.passRight
                self.round += 1
                self.lastPick = 0
                self.doneCountLock.release()
                self.startRound()
                return
            self.doneCountLock.release()


    class Player():
        def __init__(self, id, room):
            self.isRobot = False
            self.id = id
            self.room = room
            self.cards = []
            self.packQueue = []
            self.seat = -1
            self.queueLock = threading.Lock()

        def recieve(self, pack):
            self.queueLock.acquire()
            self.packQueue.append(pack)
            if len(self.packQueue) == 1:
                self.room.serve(self.id, self.packQueue[0])

            self.queueLock.release()

        def send(self, pick, nextPlayer):
            self.queueLock.acquire()
            pack = self.queue.pop(0)
            self.cards.append(pack.pick(pick))
            self.queueLock.release()

            if pack.len() == 0:
                self.room.playerFinished()
            else:
                nextPlayer.revieve(pack) #make thread

            #then serve self if queue > 1



    class RoboDrafter(Player):
        def __init__(self, room):
            super.__init__(uuid.uuid4(), room)
            self.isRobot = True

        def pick(self):
            self.room.onPick(self.id, random.randint(0, self.packQueue[0].len() - 1))


        #override receive Probably idk




    def __init__(self,startRoom,servePack):
        self.room = self.Room(servePack)
        self.players = {}
        self.startRoom = startRoom
        self.servePack = servePack


    def on_connect(self, player):
        print('connect')
        self.players[player] = self.Player(player, self.room) 
        self.room.addPlayer(self.players[player])
        self.startRoom(player)

    def on_disconnect(self, player):
        print(f'{player} disconnected')
        self.room.remove(player)
        del self.players[player]

    def startDraft(self ):
        for _ in range(len(self.room.players), self.room.MIN_PLAYERS):
            self.room.addPlayer(self.RoboDrafter(self.room))
        self.room.startRound()
        

    def on_pick(self, player, card):
        self.room.onPick(player, card)
