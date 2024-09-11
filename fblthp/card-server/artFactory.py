import requests, urllib
from bs4 import BeautifulSoup
import urllib.parse
import urllib.request
from datetime import datetime


def getGoogleArt(card):

    now = datetime.now()

    file_name = f'images/{now.strftime('%H_%M_%S_%f')}.png'

    src = googleArt(card['name'])

    urllib.request.urlretrieve(src, file_name)

    return file_name


def googleArt(name):
    query = urllib.parse.quote_plus(name)

    search_url = f"https://www.google.com/search?q={query}&tbm=isch"

    response = requests.get(search_url)

    soup = BeautifulSoup(response.text, 'html.parser')
    imgs = soup.find_all('img')
    for img in imgs:
        if not 'searchlogo' in img['src']:
            return img['src']


    return 'server/out.png'