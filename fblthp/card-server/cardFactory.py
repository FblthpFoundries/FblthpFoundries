import secret_tokens, artFactory
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading, re, json

MAX = 500
MIN = 10

def parse_card_data(input_text):
    cards = []
    
    # Define patterns for each part of the card
    card_pattern = r'<tl>(.*?)<\\tl>'
    name_pattern = r'<name>(.*?)<\\name>'
    mc_pattern = r'<mc>(.*?)<\\mc>'
    ot_pattern = r'<ot>(.*?)<\\ot>'
    power_pattern = r'<power>(.*?)<\\power>'
    toughness_pattern = r'<toughness>(.*?)<\\toughness>'
    loyalty_pattern = r'<loyalty>(.*?)<\\loyalty>'
    ft_pattern = r'<ft>(.*?)<\\ft>'
    
    # Split the input into sections for each card
    card_matches = re.findall(r'<tl>.*?(?=<tl>|$)', input_text, re.DOTALL)
    
    for card_match in card_matches:
        card = {}
        
        # Extract each component using the patterns
        card['type_line'] = re.search(card_pattern, card_match).group(1).strip()
        
        name = re.search(name_pattern, card_match)
        card['name'] = name.group(1).strip() if name else None
        
        mc = re.search(mc_pattern, card_match)
        card['mana_cost'] = mc.group(1).strip() if mc else None
        
        ot = re.search(ot_pattern, card_match)
        card['oracle_text'] = re.sub(r'<nl>', '\n', ot.group(1).strip()) if ot else None
        card['oracle_text'] = card['oracle_text'].replace('<br>', '\n')
        card['oracle_text'] = card['oracle_text'].replace('~', card['name'])
        
        power = re.search(power_pattern, card_match)
        card['power'] = power.group(1).strip() if power else None
        
        toughness = re.search(toughness_pattern, card_match)
        card['toughness'] = toughness.group(1).strip() if toughness else None
        
        loyalty = re.search(loyalty_pattern, card_match)
        card['loyalty'] = loyalty.group(1).strip() if loyalty else None
        
        ft = re.search(ft_pattern, card_match)
        card['flavor_text'] = re.sub(r'<nl>', '\n', ft.group(1).strip()) if ft else None
        
        cards.append(card)
    
    return cards

class Factory():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.model = AutoModelForCausalLM.from_pretrained('FblthpAI/magic_model', token = secret_tokens.hugging_face_token())
        self.device = 'cuda'

        self.queue = []
        self.lock = threading.Lock()
        t = threading.Thread(target=self.ensureGirth)
        t.start()

    def __populate(self, text = '<tl>'):
        encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
        maxLength = 800
        if not text == '<tl>':
            maxLength = 200
        self.model.to(self.device)
        output = self.model.generate(
            **encoded_input,
            do_sample=True,
            temperature = 0.9,
            max_length =maxLength,
        )

        cards = self.tokenizer.batch_decode(output)[0].split('<eos>')
        cards = parse_card_data(''.join(cards[:-1]))
        for card in cards:
            self.queue.append((card, artFactory.getGoogleArt(card)))

        if len(cards) == 0:
            self.__populate(text)

    def ensureGirth(self,):
        while len(self.queue) > MAX:
            self.lock.acquire()
            self.queue().pop()
            self.lock.release()

        while len(self.queue) < MIN:
            self.lock.acquire()
            self.__populate() 
            self.lock.release()




    def consume(self, text = None):
        card = {}
        self.lock.acquire()
        if text:
            length = len(self.queue)
            self.__populate(text)
            card = self.queue.pop(length)
        else:
            card =  self.queue.pop()

        self.lock.release()

        t = threading.Thread(target=self.ensureGirth)
        t.start()

        return card[0], card[1]

