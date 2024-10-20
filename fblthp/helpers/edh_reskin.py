from pathlib import Path
import requests
import openai
import random
import json
from constants import API_KEY
openai.api_key = API_KEY


PROMPTS_DIR = Path(__file__).parent / "prompts"

def get_multiline_input(prompt):
    print(prompt)
    lines = []
    blanks = 0
    while True:
        line = input()
        if line:  # Continue until an empty line is entered
            lines.append(line)
            blanks = 0
        else:
            blanks += 1
            if blanks == 2:
                break
    return "\n".join(lines)
def ask_gpt(prompt, header="You are an expert in Magic: The Gathering.", snatch_json=True):
        #print(f"ME: {prompt}")
        
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": header},
                {"role": "user", "content": prompt}
            ],
        )
        resp = response.choices[0].message.content.strip()
        if snatch_json:
            if '```' in resp:
                resp = resp.split('```')[1]
            if resp[:4] == "json":
                resp = resp[4:]
            #print(f"CHATGPT: {resp}")
            try:
                resp = json.loads(resp)
                return True, resp
            except:
                return False, None
        else:
            return resp
def grab_card(card_name):
    # Replace spaces with '+' for the API request
    card_name = card_name.replace(' ', '+')
    url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        card_data = response.json()
        return card_data
    else:
        return None
    

def deck_pipeline(deck, sample_size=5, output_file="output.txt"):
    colors = {
        "W": "white",
        "U": "blue",
        "B": "black",
        "R": "red",
        "G": "green"
    }
    format = "MTGO"
    
    with open(output_file, "w") as f_out:
        if format == "MTGO":
            deck = deck.split("\n")
            deck = [(int(x), y) for x, y in [x.strip().split(" ", 1) for x in deck if len(x.strip()) > 0]]
            commander = grab_card(deck[-1][1])
            with open(PROMPTS_DIR / "commander_query.txt", "r") as f:
                    prompt = f.read().format(
                        commander=commander['name']
                    )
            summary = ask_gpt(prompt, snatch_json=False)
            for card in deck[:-1]:
                with open(PROMPTS_DIR / "edh_reskin.txt", "r") as f:
                    scard = grab_card(card[1])
                    ci = [colors[c] for c in scard['color_identity']]
                    if not ci:
                        ci = "Colorless, Devoid, Gray"
                    else:
                        ci = ", ".join(ci)

                    prompt = f.read().format(
                        commander_name=commander['name'],
                        card=scard['name'],
                        summary=summary,
                        color_identity=ci,
                        additional="",
                        sample_size=sample_size,
                    )
                    res = ask_gpt(prompt, snatch_json=True)
                    if res[0]:
                        res = res[1]
                        f_out.write(scard['name'] + "\n")
                        for p in res['prompts']:
                            f_out.write(p + "\n\n")
                        f_out.write("\n")
                    else:
                        f_out.write("Failed to generate prompts for " + scard['name'] + "\n\n")


if __name__ == "__main__":
    #deck = get_multiline_input("Enter your decklist:")
    deck = '''
    1 All That Glitters
1 Angelic Destiny
1 Angelic Gift
1 Anguished Unmaking
1 Animate Dead
1 Arcane Lighthouse
1 Arcane Signet
1 Ardenn, Intrepid Archaeologist
1 Ashiok's Reaper
1 Black Market Connections
1 Bloodthirsty Blade
1 Brightclimb Pathway
1 Caves of Koilos
1 Codsworth, Handy Helper
1 Command Tower
1 Concealed Courtyard
1 Dark Ritual
1 Darksteel Mutation
1 Detection Tower
1 Eidolon of Countless Battles
1 Eldrazi Conscription
1 Ethereal Armor
1 Faithbound Judge
1 Feather of Flight
1 Fellwar Stone
1 Fetid Heath
1 Generous Gift
1 Ghoulish Impetus
1 Gift of Immortality
1 Glasswing Grace
1 Glistening Oil
1 Godless Shrine
1 Gryff's Boon
1 Hall of Heliod's Generosity
1 Hallowed Haunting
1 Hateful Eidolon
1 Hero of Iroas
1 Isolated Chapel
1 Kaya's Ghostform
1 Killian, Ink Duelist
1 Kor Spiritdancer
1 Light-Paws, Emperor's Voice
1 Lightning Greaves
1 Lord Skitter's Blessing
1 Martial Impetus
1 Mesa Enchantress
1 Nurgle's Rot
1 On Thin Ice
1 Ondu Spiritdancer
1 Open the Armory
1 Orzhov Signet
1 Ossification
1 Parasitic Impetus
1 Pearl-Ear, Imperial Advisor
1 Phyresis
1 Redemption Arc
1 Reprobation
1 Resurgent Belief
1 Retether
1 Sage's Reverie
1 Shattered Sanctum
1 Sheltered by Ghosts
1 Sigarda's Aid
1 Sigil of the Empty Throne
1 Silent Clearing
1 Smothering Tithe
8 Snow-Covered Plains
7 Snow-Covered Swamp
1 Sol Ring
1 Songbirds' Blessing
1 Sphere of Safety
1 Spirit Mantle
1 Sram, Senior Edificer
1 Starfield Mystic
1 Swiftfoot Boots
1 Swords to Plowshares
1 Tainted Field
1 Talisman of Hierarchy
1 Temple of Silence
1 Timely Ward
1 Transcendent Envoy
1 Unquestioned Authority
1 Vault of Champions
1 Well of Lost Dreams
1 Winds of Rath

1 Eriette of the Charmed Apple
    '''
    deck_pipeline(deck)