from artFactory import ArtGen
from cardGenerator import CardGen
import logging
import threading

MAX = 500
MIN = 10

class Factory():
    def __init__(self, cardGen: CardGen, artGen: ArtGen, logger: logging.Logger ):
        self.cardGen = cardGen
        self.artGen = artGen
        self.logger = logger

        self.queue = []
        self.cardLock = threading.Lock()
        self.girthLock = threading.Lock()
        t = threading.Thread(target=self.ensureGirth)
        t.start()

    def __populate(self):

        cards = self.cardGen.generate()

        self.artGen.getArt(cards)

        images = self.artGen.renderBatch(cards)

        for card in images:
            self.cardLock.acquire()
            self.queue.append(card)
            self.cardLock.release()
        
        if len(cards) == 0:
            self.__populate()

    def ensureGirth(self,):
        self.girthLock.acquire()
        while len(self.queue) > MAX:
            self.cardLock.acquire()
            self.queue.pop()
            self.cardLock.release()

        while len(self.queue) < MIN:
            self.__populate() 

        self.girthLock.release()




    def consume(self):
        card = {}
        self.cardLock.acquire()
        card =  self.queue.pop()
        self.cardLock.release()

        t = threading.Thread(target=self.ensureGirth)
        t.start()

        return card

