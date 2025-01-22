from multiprocessing import Process
from flask import Flask, request, jsonify
import  cardFactory, cardGenerator, artFactory
import json, logging, sys


def create_app():
    app = Flask(__name__)
    logging.basicConfig(filename='log.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.info('Start')
    cardGen = cardGenerator.GPT2Gen(logger)
    artGen = artFactory.ArtGen(logger)
    factory = cardFactory.Factory(cardGen, artGen, logger)

    @app.route('/')
    def hello():
        return 'sup bitches'
    
    """
    @app.route('/make_card', methods = ['POST'])
    def makeCard():

        args = request.get_json()

        if not 'text' in args:
            return 'fuck you bitch', 400

        text = None if args['text'] == '' else args['text'].replace('/', '\\')

        card, art = factory.consume(text)
        
        card = htmlRender.renderCard(card, art)

        return jsonify({'card':card})
    

    @app.route('/make_many_cards', methods = ['POST'])
    def makeManyCards():
        args = request.get_json()

        if not 'num' in args:
            return 'fuck you bitch', 400
        
        num = args['num']

        for i in range(num):
            factory.consume()

        return jsonify({'cards':'card'})
    """
    

    @app.route('/test', methods = ['GET'])
    def test():
        return jsonify({'card':factory.consume()[0]})
    
    return app

if __name__ == '__main__':
    app = create_app()
    server = Process(app.run(host='0.0.0.0', port = 5001, debug = True))
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

    