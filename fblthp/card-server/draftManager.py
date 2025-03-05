from Pack import getSet, Pack, getSetList
from packQueue import PackQueue
import uuid, random, threading
class DraftManager():
    class Room():

        MIN_PLAYERS = 4
        NUM_ROUNDS=1
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


        """
        sets up round by setting each player's next player
        and serving a pack to each
        """
        def startRound(self,):
            playerCount = len(self.players)
            for i in range(playerCount):
                nextIdx = i + 1 if self.passRight else i -1
                nextIdx %= playerCount
                self.players[i].nextPlayer = self.players[nextIdx]

            for p in self.players:
                p.recievePack(Pack(self.set))

        """
        Serves packs to players, both robot and mortally challenged
        """
        def serve(self, player, pack):
            if player.isRobot:
                roboPick = threading.Thread(player.recievePack(pack))
                roboPick.start()
            else:
                self.servePack(player.id, pack.toJson())
        """
        After recieving a pick from client, updates internal state
        appropriately by picking card from pack for the player
        """
        def onPick(self, playerId, pick):
            player = None
            #find player based on ID
            for p in self.players:
                player = p
                if p.id == playerId:
                    break
        
            player.pickPack(pick)

            self.doneCountLock.acquire()

            if self.lastPick == len(self.players):
                if self.round == self.NUM_ROUNDS:

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
            self.packQueue = PackQueue()
            self.seat = -1
            self.nextPlayer = None

        """
        Adds pack to queue and serves pack to player if player waiting for pack
        """
        def recievePack(self, pack):
            send = self.packQueue.push(pack)
            if send: #new curPack so serve to player
                self.room.serve(self, self.packQueue.peek())

        def pickPack(self, pick):
            card, pack, havePack = self.packQueue.pick(pick)
            self.cards.append(card)

            if pack.len() == 0:
                self.room.playerFinished()
                return
            if havePack:
                self.room.serve(self, self.packQueue.peek())  
            self.nextPlayer.recievePack(pack) 



    class RoboDrafter(Player):
        def __init__(self, room):
            super().__init__( uuid.uuid4(), room)
            self.isRobot = True



        def recievePack(self, pack):
            pick = self.packQueue.push(pack)
            print(pick)
            if pick: #new curPack so pick
                cardNum = self.packQueue.peek().len()
                pickThread = threading.Thread(self.pickPack(random.randint(0, cardNum - 1)))
                pickThread.start()
                

                


    def __init__(self,startRoom,servePack):
        self.room = self.Room(servePack)
        self.players = {}
        self.startRoom = startRoom
        self.servePack = servePack
        self.sets = getSetList()

    def getSets(self,):
        return self.sets 


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
