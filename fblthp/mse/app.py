from flask import Flask


def createApp():
    app = Flask(__name__)

    @app.route('/')
    def default():
        return 'hello world\n'


    return app

if __name__ == '__main__':
    createApp().run(host = '0.0.0.0', port = 6969)
