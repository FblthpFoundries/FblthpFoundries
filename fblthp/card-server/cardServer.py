from multiprocessing import Process
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, send, emit 
import  cardFactory, cardGenerator, artFactory
import json, logging, sys, os, base64
from draftManager import DraftManager


socketio = SocketIO(logger = True, engineio_logger = True)



def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'Fblthp\'s Balls'
    logging.basicConfig(filename='log.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.info('Start')


    socketio.init_app(app, cors_allowed_origins="*")
    draftManager = DraftManager(
        startRoom = lambda player: socketio.emit('roomStarted', room = player),
        servePack = lambda player, pack :socketio.emit('pack', pack, room = player)
        )
    '''
    cardGen = cardGenerator.GPT2Gen(logger)
    artGen = artFactory.ArtGen(logger)
    factory = cardFactory.Factory(cardGen, artGen, logger)


    def getCard():
        img = factory.consume()
        encoded = base64.b64encode(open(img, 'rb').read()).decode('utf-8')
        os.remove(img)
        return f'data:image/png;base64,{encoded}'

    @app.route('/')
    def hello():
        return 'sup bitches'
    
    @app.route('/test', methods = ['GET'])
    def test():
        img = getCard()
        return  render_template('simple.html', image = img)

    '''

    @app.route('/startDraft')
    def testPack():
        draftManager.startDraft()
        return 'pack'
    
    @socketio.on('connect')
    def connect():
        draftManager.on_connect(request.sid)

    @socketio.on('disconnect')  
    def disconnect():
        draftManager.on_disconnect(request.sid)

    @socketio.on('pick')
    def onPick(card):
        draftManager.on_pick(request.sid, card)
    
 

    return app

if __name__ == '__main__':
    app = create_app()
    socketio.run(app, host='0.0.0.0', port = 5001, debug = True)
    """
    server = Process()
    server.start()

    print('press escape to close')

    while True:
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            print('exit? [Y/N]')
            ch = sys.stdin.read(1)
            if ch == 'y' or ch == 'Y':
                break


    server.terminate()
    server.join()
    print('goodbye')

    """

    