import requests
from bs4 import BeautifulSoup
import json

url = 'https://tryhardguides.com/nyt-connections-answers/'
headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36'}
response = requests.get(url, headers=headers)
# print(response.content.decode())
cols = {0: 'yellow', 1: 'green', 2: 'blue', 3: 'purple'}

if response.status_code == 200:
    web_content = response.content
    
    soup = BeautifulSoup(web_content, 'html.parser')
    connections = {}

    outer_li_tags = soup.find_all('li')
    day = 0

    for outer_li in outer_li_tags:
        nested_ul = outer_li.find('ul')
        if nested_ul:
            nested_li_tags = nested_ul.find_all('li')
            colours = {}
            difficulty = 0

            for nested_li in nested_li_tags:
                category_tag = nested_li.find('strong')
                if category_tag:
                    category = category_tag.get_text()
                    words = nested_li.get_text().replace(category, '', 1).replace('-', '').strip()
                    category = ' '.join([c for c in category.split() if c.isalpha()])
                    colours[cols[difficulty]] = {category: [word.strip() for word in words.split(',')]}
                    difficulty += 1

            connections[day] = colours
            day += 1

    with open('data.json', 'w') as file:
        json.dump(connections, file, indent=4)
else:
    print(f"Failed to retrieve the web page. Status code: {response.status_code}")