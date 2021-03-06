import requests as req
from googleapiclient.discovery import build
import json
from bs4 import BeautifulSoup

# Google search using api
def google_search(search_term, api_key, cse_id):
    service = build("customsearch", "v1",
            developerKey=api_key)

    res = service.cse().list(
      q=search_term,
      cx=cse_id,
    ).execute()
    return res
 
# Takes a sentence and returns a json from isbndb   
def run(sent):
    # Keys
    f = open("../data/se_id.txt", "r")
    se_id = f.read()
    f = open("../data/api_key.txt", "r")
    api_key = f.read()
    
    # Get link
    result = google_search(sent, api_key, se_id)
    url = ''
    for element in result['items']:
        link = element['link']
        if 'goodreads.com' in link:
            url = link
            break
    
    # Break out if not found
    if url == '':
        print('FAILURE 1') #TODO: Make it better
        return 'FAIL'
    
    isbn = ''
    page = req.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    nums = soup.find_all('span', itemprop='isbn')
    isbn = nums[0].get_text()
    print(isbn)
    '''
    for n in nums:
        nu = str(n)
        if len(nu) == 10:
            isbn = n '''
    
    # Break out if not found
    if isbn == '':
        print('FAILURE 2') #TODO: Make it better
        return 'FAIL'
    
    # Authorise
    f = open("../data/rest_key.txt", "r")
    key = f.read()
    h = {'Authorization': key}
    
    # Prepare request
    #resp = req.get("https://api2.isbndb.com/book/9781934759486", headers=h)
    request = 'https://api2.isbndb.com/book/' + isbn
    
    resp = req.get(request, headers=h)
    return (resp.json(), isbn)

