import requests
from bs4 import BeautifulSoup
import json

url = "https://docs.python.org/3/faq/general.html#general-python-faq"

response = requests.get(url)
html_content = response.text

soup = BeautifulSoup(html_content, 'html.parser')

# Remove ''contents' div
contents_nav = soup.find('nav', class_='contents')
if contents_nav:
    contents_nav.decompose()

# Remove 'sphinxsidebarwrapper' div
sidebar_div = soup.find('div', class_='sphinxsidebarwrapper')
if sidebar_div:
    sidebar_div.decompose()

# Dictionary to hold our extracted information
data = []

# Iterate over the elements, extracting the required information
for i in range(4, 28):
    element = soup.find(href=f"#id{i}")
    if element:
        title = element.get_text(strip=True)
        content_paragraphs = element.find_all_next('p', limit=2)
        content = " ".join(p.get_text(strip=True) for p in content_paragraphs)
        
        data.append({"title": title, "content": content})


# Convert the dictionary to a JSON string
json_data = json.dumps(data, indent=4)

with open('faq_data.json', 'w', encoding='utf-8') as f:
    f.write(json_data)       
