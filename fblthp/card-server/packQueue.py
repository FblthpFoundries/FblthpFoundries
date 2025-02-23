from Pack import Pack
import threading

class PackQueue():
    def __init__(self):
        self.queue = []
        self.curPack = None
        self.queueLock = threading.Lock()

    def peek(self):
        return self.curPack
    
    def pick(self, pickId):
        card = self.curPack.pick(pickId)

        passPack = self.curPack
        self.curPack = None

        self.queueLock.acquire()
        if len(self.queue) > 0:
            self.curPack = self.queue.pop(0)
        self.queueLock.release()

        return card, passPack, not self.curPack == None
    
    def push(self, pack):
        sendPack = False #send pack if no curPack
        self.queueLock.acquire()
        idx = 0
        for i in range(len(self.queue)):
            if pack.len() > self.queue[i].len():
                break
            idx += 1
        self.queue.insert(idx, pack)
        if not self.curPack:
            self.curPack = self.queue.pop(0)
            sendPack = True
        self.queueLock.release()

        return sendPack
    
    def len(self):
        length = 1 if self.curPack else 0
        self.queueLock.acquire()
        length += len(self.queue)
        self.queueLock.release()
        return length
