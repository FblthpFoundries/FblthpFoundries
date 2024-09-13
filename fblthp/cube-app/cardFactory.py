from transformers import AutoTokenizer, AutoModelForCausalLM
import re, threading
import secret_tokens
from magicCard import Card

class Factory():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.model = AutoModelForCausalLM.from_pretrained('FblthpAI/magic_model', token = secret_tokens.hugging_face_token())
        self.device = 'cuda'
        self.model.to(self.device)
        self.cards = []

    def __cardGen(self, updateFunc = None):
        text = '<tl>'
        encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model.generate(
            **encoded_input,
            do_sample=True,
            temperature = 0.9,
            max_length =400,
        )
        cardTokens = self.tokenizer.batch_decode(output)[0].split('<eos>')
        cardTokens = self.parse_card_data(''.join(cardTokens[:-1]))


        for card in cardTokens:
            if not self.is_valid_card(card):
                pass
            self.cards.append(Card(card))
            if updateFunc:
                updateFunc(len(self.cards))

    def gen_cube(self, num, update):

        while len(self.cards) < num:
            self.__cardGen(lambda n : update(n/num))

        toReturn = self.cards[:num]
        self.cards = self.cards[num:]


        return toReturn
        

    def reroll(self):
        if len(self.cards) > 0:
            return self.cards.pop()

        self.__cardGen()        

        return self.cards.pop()

    def is_valid_card(self,card):
        if not 'name' in card:
            return False
        if not 'oracle_text' in card:
            return False
        if not 'type_line' in card:
            return False
        return True

    def parse_card_data(self, input_text):
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