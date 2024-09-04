import imgkit

def renderCard(card, art):
    options = {
        'format': 'png',
        'enable-local-file-access': ''
    }
    body = """
    <html>
        <head>
            <style>
                @font-face{
                    font-family:"name-font";
                    src: url('file:///C:/Users/oweng/magic/fblthp/card-server/fonts/MagictheGathering.TTF');
                }
                @font-face{
                    font-family:"mana-font";
                    src: url('file:///C:/Users/oweng/magic/fblthp/card-server/fonts/Proxyglyph.TTF');
                }
                body{
                    height: 1042px;
                    width 750px;
                }
                .card{
                    background-image: url('file:///C:/Users/oweng/magic/fblthp/card-server/card_templates/Black_Template.png');
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
            </style>
        </head>
        <body>
            <div class="card">

                <div class="name">
                    Recurring Nightmare
                </div>
                <div class="mana">
                    <span style="color: #848484">Q</span>2 <span style="color: #848484">Q</span>b
                </div>
                <div class="imgae>
                </div>
                <div class="oracle_text">
                </div>
                <div class="flavor_text">
                </div>
                <div class="pt">
                </div>


            </div>
        </body>
    </html>
    """

    imgkit.from_string(body, 'out.png', options=options)

    return

if __name__ == '__main__':
    card = {"flavor_text":"In the face of overwhelming odds, goblin shamans always succeed.","loyalty":"","mana_cost":"<4> <R>","name":"Goblin Looter","oracle_text":"This is a big huge test \n TEST","power":"4","toughness":"4","type_line":"Creature - Goblin Rogue"}
    renderCard(card, 'picture.jpg') 