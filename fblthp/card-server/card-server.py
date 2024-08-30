from flask import Flask
from transformers import AutoTokenizer, AutoModelForCausalLM
import secret_tokens

def create_app():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('FblthpAI/magic_model', token = secret_tokens.hugging_face_token())
    app = Flask(__name__)
    device = 'cuda'

    @app.route('/')
    def hello():
        return 'sup bitches'

    @app.route('/make_card', methods = ['GET'])
    def makeCard():
        text = '<tl>'
        encoded_input = tokenizer(text, return_tensors='pt').to(device)
        model.to(device)
        output = model.generate(
            **encoded_input,
            do_sample=True,
            temperature = 0.9,
            max_length =200,
        )

        return tokenizer.batch_decode(output)[0].split('<eos>')[0].replace('—', '-').replace('\u2212', '-')
    
    return app

if __name__ == '__main__':
    create_app.run( debug = True)