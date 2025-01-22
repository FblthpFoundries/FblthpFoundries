from card import Card
import logging
import re,random

class CardGen:
    def __init__(self, logger: logging.Logger ):
        self.logger = logger
        pass
    def generate(self) -> list[Card]:
        pass

class GPT2Gen(CardGen):
    def __init__(self, logger):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from secret_tokens import hugging_face_token
        import torch
        super().__init__(logger)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        print(f"Loaded tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained('FblthpAI/magic_model', token = hugging_face_token())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.model.to(self.device)
        self.batchSize = 10

    def generate(self) -> list[Card]:
        text = ['<tl>'] * self.batchSize
        encoded_input = self.tokenizer(text, return_tensors='pt').to(self.device)
        output = self.model.generate(
            **encoded_input,
            do_sample=True,
            temperature = 0.9,
            max_length =400,
            pad_token_id=self.tokenizer.eos_token_id
        )

        cardOutput = self.tokenizer.batch_decode(output)
        cardTokens = ''
        #collect all but last card from each batch
        #last card often is cut off by max_length
        for batch in cardOutput:
            for card in batch.split('<eos>')[:-1]:
                cardTokens += card.replace('\u2212', '-').replace('\u2014', '-').replace('\u2022', '')

        cardDicts = self._parse_card_data(cardTokens)

        cards = []

        for card in cardDicts:
            if not self._is_valid_card(card):
                pass
            cards.append(Card(card))

        return cards

        

    def _parse_card_data(self, input_text: str) -> list[dict]:
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
            if not re.search(card_pattern, card_match):
                continue
            card['type_line'] = re.search(card_pattern, card_match).group(1).strip()
            
            name = re.search(name_pattern, card_match)
            card['name'] = name.group(1).strip() if name else None
            
            mc = re.search(mc_pattern, card_match)
            card['mana_cost'] = mc.group(1).strip() if mc else None
            
            ot = re.search(ot_pattern, card_match)
            card['oracle_text'] = re.sub(r'<nl>', '\n', ot.group(1).strip()) if ot else None
            if not card['oracle_text'] :
                continue
            card['oracle_text'] = card['oracle_text'].replace('<br>', '\n')
            if not card['name']:
                continue
            card['oracle_text'] = card['oracle_text'].replace('~', card['name'])
            
            power = re.search(power_pattern, card_match)
            card['power'] = power.group(1).strip() if power else None
            
            toughness = re.search(toughness_pattern, card_match)
            card['toughness'] = toughness.group(1).strip() if toughness else None
            
            card["rarity"] = random.choices(["common", "uncommon", "rare", "mythic rare"], [0.1, 0.25, 0.5, 0.15])[0]
            
            loyalty = re.search(loyalty_pattern, card_match)
            card['loyalty'] = loyalty.group(1).strip() if loyalty else None
            
            ft = re.search(ft_pattern, card_match)
            card['flavor_text'] = re.sub(r'<nl>', '\n', ft.group(1).strip()) if ft else None
            
            cards.append(card)

        return cards   

    def _is_valid_card(self,card):
        if not 'name' in card:
            return False
        if not 'oracle_text' in card:
            return False
        if not 'type_line' in card:
            return False
        return True