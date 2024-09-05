import imgkit
import re

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
                    background-image: url('file:///server/card_templates/Black_Template.png');
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
                img{
                    width:570px;
                    height:465px;
                    object-fit:fill;
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
                    <img src="file:///server/picture.jpg" />
                </div>
                <div class="type_line">
                    TL
                </div>
                <div class="set_symbol">
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

    html = updateBody(card)
    

    imgkit.from_string(html, 'out.png', options=options)

    return html

mana_colors={
    'W':'#fcfcc1',
    'B':'#848484',
    'U':'#67c1f5',
    'R':'#f85555',
    'G':'#26b569',
}


def updateManaCost(string):
    single_digit = r'<[0-9]>'

    to_replace = re.findall(single_digit, string)

    for mana in to_replace:
        string = string.replace(mana,  f'<span style=\"color: #848484\">Q</span>{mana[1]}')


    single_mana = r'<[WUBRG]>'

    to_replace = re.findall(single_mana, string)

    for mana in to_replace:
        string = string.replace(mana,  f'<span style=\"color:{mana_colors[mana[1]]} \">Q</span>{mana[1].lower()}')


    single_special = r'<[TX]>'
    
    to_replace = re.findall(single_special, string)

    for mana in to_replace:
        char = 'X' if mana[1] == 'X' else 't'
        string = string.replace(mana,  f'<span style=\"color: #848484\">Q</span>{char}')


    return string

def updateBody(card):
    html = body.replace('NAME', card['name'])
    html = html.replace('MC', updateManaCost(card['mana_cost']))
    html = html.replace('TL', card['type_line'])
    html = html.replace('OT', updateManaCost(card['oracle_text']).replace('\n','<br/>'))
    html = html.replace('FT', card['flavor_text'].replace('\n','<br/>'))
    html = html.replace('PT', f'{card['power']}/{card['toughness']}' if card['power'] else '')
    return html


if __name__ == '__main__':
    card = {"flavor_text":"In the face of overwhelming odds, goblin shamans always succeed.","loyalty":"","mana_cost":"<4> <R>","name":"Goblin Looter","oracle_text":"This is a big huge test \n TEST","power":"4","toughness":"4","type_line":"Creature - Goblin Rogue"}
    renderCard(card, 'picture.jpg') 