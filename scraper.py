import requests
from bs4 import BeautifulSoup

url = 'https://tryhardguides.com/nyt-connections-answers/'
headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36'}
response = requests.get(url, headers=headers)
# print(response.content.decode())

if response.status_code == 200:
    web_content = response.content
    
    soup = BeautifulSoup(web_content, 'html.parser')

    titles = soup.find_all('li', class_='')
    with open('scrape_results.txt', 'w') as f:
        for index, title in enumerate(titles):
            f.write(f"{title.text}\n")
else:
    print(f"Failed to retrieve the web page. Status code: {response.status_code}")