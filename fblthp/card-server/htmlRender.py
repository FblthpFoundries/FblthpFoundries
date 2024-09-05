import imgkit
import re
import base64
from datetime import datetime
import os

to_absolute = 'C:/Users/oweng/magic/fblthp/card-'

body = """
    <html>
        <head>
            <style>
                @font-face{
                    font-family:"name-font";
                    src: url('file:///server/fonts/MagictheGathering.TTF');
                }
                @font-face{
                    font-family:"mana-font";
                    src: url('file:///server/fonts/Proxyglyph.TTF');
                }
                @font-face{
                    font-family:"body-font";
                    src: url('file:///server/fonts/PlantinMTProRg.TTF');
                }
                @font-face{
                    font-family:"flavor-font";
                    src: url('file:///server/fonts/PlantinMTProRgIt.TTF');
                }
                @font-face{
                    font-family:"reminder-font";
                    src: url('file:///server/fonts/PlantinMTProSemiBdIt.TTF');
                }
                body{
                    height: 1042px;
                    width 750px;
                }
                .card{
                    background-image: url('file:///CARDBACKGROUND');
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                    background-size: auto;
                    height: 1042px;
                    width 750px;
                }
                .name{
                    font-family:"name-font";
                    position:absolute;
                    color: white;
                    top:55px;
                    left:80px;
                    font-size: 35px;
                    width:500px;
                    height:50;
                    text-align:justify;
                    text-shadow: 2px 2px black;
                }
                .mana{
                    font-family:"mana-font";
                    position:absolute;
                    color:black;
                    top:55px;
                    left:600px;
                    width:150px;
                    height:50px;
                    font-size:25px;
                }
                .type_line{
                    font-family:"body-font";
                    position:absolute;
                    color:white;
                    top:580px;
                    left:80px;
                    font-size:25px;
                    text-shadow: 2px 2px black;
                }
                .body_text{
                    position:absolute;
                    color:black;
                    left:90px;
                    top:640px;
                    width:555px;
                    height:260px;
                    font-size:25px;
                }
                .oracle_text{
                    font-family:"body-font";
                    text-align:justify;
                    height:auto;
                }
                .flavor_text{
                    font-style:italic;
                    font-family:"flavor-font";
                    height:auto;
                    text-align:right;
                    margin-top:10px;
                }
                .image{
                    position:absolute;
                    top:100px;
                    left:85px;
                    width:570px;
                    height:465px;
                }
                .pt{
                    position:absolute;
                    width:120px;
                    height:60px;
                    left:590px;
                    top:940px;
                    color:white;
                    font-size:50px;
                    font-family:"body-font";
                    text-shadow: 2px 2px black;
                }
                .set_symbol{
                    position:absolute;
                    top:580px;
                    left:610px;
                }
                #mana_cost{
                    font-family:"mana-font";
                }
                
            </style>
        </head>
        <body>
            <div class="card">
                <div class="name">
                   NAME
                </div>
                <div class="mana">
                    MC
                </div>
                <div class="image">
                    <img src="file:///server/picture.jpg" style="width:570px;height:465px;object-fit:fill;" />
                </div>
                <div class="type_line">
                    TL
                </div>
                <div class="set_symbol">
                    <img src="file:///server/Logo.jpg" style="width:40px;height:40px;object-fit:fill"/>
                </div>
                <div class="body_text">
                    <div class="oracle_text">
                        OT
                    </div>
                    <div class="flavor_text">
                       FT 
                    </div>
                </div>
                <div class="pt">
                    PT
                </div>
            </div>
        </body>
    </html>
    """

def renderCard(card, art):
    options = {
        'format': 'png',
        'enable-local-file-access': ''
    }

    now = datetime.now()

    file_name = f'{now.strftime('%H_%M_%S_%f')}.png'

    html = updateBody(card)
    
    imgkit.from_string(html, file_name, options=options)

    encoded = base64.b64encode(open(file_name, 'rb').read())

    os.remove(file_name)

    return b'data:image/png;base64,' + encoded

mana_colors={
    'W':'#fcfcc1',
    'B':'#848484',
    'U':'#67c1f5',
    'R':'#f85555',
    'G':'#26b569',
}

color_backgrounds={
    'W':'White_Template.png',
    'B':'Black_Template.png',
    'U':'Blue_Template.png',
    'R':'Red_Template.png',
    'G':'Green_Template.png',
    'MC':'MultiColored_Template.png',
    'C':'Colorless_Template.png',
}

def updateManaCost(string):
    start_font = '<span id=\"mana_cost\">'
    end_font = '</span>'
    single_digit = r'<[0-9]>'

    to_replace = re.findall(single_digit, string)

    for mana in to_replace:
        string = string.replace(mana,  f'{start_font}<span style=\"color: #848484;\">Q</span>{mana[1]}{end_font}')


    single_mana = r'<[WUBRG]>'

    to_replace = re.findall(single_mana, string)

    for mana in to_replace:
        string = string.replace(mana,  f'{start_font}<span style=\"color:{mana_colors[mana[1]]};\">Q</span>{mana[1].lower()}{end_font}')


    single_special = r'<[TX]>'
    
    to_replace = re.findall(single_special, string)

    for mana in to_replace:
        char = 'X' if mana[1] == 'X' else 't'
        string = string.replace(mana,  f'{start_font}<span style=\"color: #848484;\">Q</span>{char}{end_font}')


    return string

def setBackground(card, body):
    color_regex = r'[WUBRG]'

    colors = re.findall(color_regex, card['mana_cost'])
    card_color = 'C' if len(colors) == 0 else colors[0]

    for color in colors:
        if not color == card_color:
            card_color = 'MC'
            break

    return body.replace('CARDBACKGROUND', f'server/card_templates/{color_backgrounds[card_color]}')

    

def updateBody(card):
    html = setBackground(card, body)
    html = html.replace('NAME', card['name'])
    html = html.replace('MC', updateManaCost(card['mana_cost']))
    html = html.replace('TL', card['type_line'])
    html = html.replace('OT', updateManaCost(card['oracle_text']).replace('\n','<br/>'))
    html = html.replace('FT', card['flavor_text'].replace('\n','<br/>'))
    html = html.replace('PT', f'{card['power']}/{card['toughness']}' if card['power'] else '')
    return html


if __name__ == '__main__':
    card = {"flavor_text":"In the face of overwhelming odds, goblin shamans always succeed.","loyalty":"","mana_cost":"<U> <W>","name":"Goblin Looter","oracle_text":"This is a big huge test \n TEST \n<X> <T>: Kill X cards","power":"4","toughness":"4","type_line":"Creature - Goblin Rogue"}
    renderCard(card, 'picture.jpg') 