from gpt import gen
import argparse
#from bs4 import BeautifulSoup
import re

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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Magic the Gathering cards')
    parser.add_argument('--text_start', type=str, default='<tl>', help='The text to start the generation with')
    parser.add_argument('--max_length', type=int, default=400, help='The maximum length of the generated text')
    args = parser.parse_args()
    output = gen(
        text_start=args.text_start,
        max_length=args.max_length,
        model_path='./magic_mike/checkpoint-8500'
    )
    #output = "<tl> Creature — Vampire Rogue <\\tl> <name> Vengeful Rogue <\\name> <mc> <3> <B> <\\mc> <ot> <T>: Draw a card, then discard a card. <\\ot> <power> 4 <\\power> <toughness> 4 <\\toughness> <loyalty> <\\loyalty> <ft> The greatest evil lies behind all the shadows and shadows in my mind, and all I fear is not death, but only peace. <nl> —The Dark Path of the Eternals <\\ft><eos><tl> Creature — Bird Soldier <\\tl> <name> Gullwing Pilot <\\name> <mc> <4> <W> <\\mc> <ot> Flying <nl> Whenever ~ deals combat damage to a player, destroy target nonland permanent. <\\ot> <power> 3 <\\power> <toughness> 3 <\\toughness> <loyalty> <\\loyalty> <ft> When not flying, he rides to the skies. <nl> —Oedipus <\\ft><eos><tl> Artifact <\\tl> <name> Sword of the Eternals <\\name> <mc> <3> <\\mc> <ot> <T>: Target creature you control gets +2/+2 until end of turn. <\\ot> <power> <\\power> <toughness> <\\toughness> <loyalty> <\\loyalty> <ft> This sword is the first gift we have given to ourselves in this world. <\\ft><eos><tl> Legendary Creature — Shapeshifter <\\tl> <name> Chandra, Witch's Apprentice <\\name> <mc> <X> <U> <B> <G"
    #print(output)
    #bs = BeautifulSoup(output, 'lxml')
    #print(bs.get_text())
    parsed = parse_card_data(output.split("<eos>")[0])[0]

    [print(f"{key}: {value}") for key, value in parsed.items()]