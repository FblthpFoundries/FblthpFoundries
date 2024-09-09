from flask import Flask, request, jsonify
import htmlRender, cardFactory
import json


def create_app():
    app = Flask(__name__)
    factory = cardFactory.Factory()

    @app.route('/')
    def hello():
        return 'sup bitches'
    

    @app.route('/make_card', methods = ['POST'])
    def makeCard():

        args = request.get_json()

        text = None if args['text'] == '' else args['text']
        
        card = htmlRender.renderCard(factory.consume(text), 'picture.jpg')

        return jsonify({'card':card})
    

    @app.route('/test', methods = ['GET'])
    def test():
        return jsonify({'card':factory.consume()})
    
    return app

if __name__ == '__main__':
    create_app.run( debug = True)