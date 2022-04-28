from bs4 import BeautifulSoup
import urllib
import urllib.request
import re
import pandas as pd
import numpy as np
import os
from datetime import datetime
import requests
import csv
import textblob
import matplotlib.pyplot as plt
import seaborn as sns


headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}

url = "https://forums.hardwarezone.com.sg/eat-drink-man-woman-16/%5Bbionix-accident%5D-second-photo-accident-leaked-5931722.html"
    
resp = requests.get(url, headers=headers)
content = resp.text
soup = BeautifulSoup(content, "lxml")
letters = soup.find_all("div", attrs={"id": re.compile ("^post_message_\d+")})
print(letters)

#Choose the main site URL
site_url = 'https://forums.hardwarezone.com.sg'

# Get the threads

def getPosts(thread_url):
    #print(thread_url)
    lastThreadPage = False
    thread_cols = ['thread_url', 'userid', 'timestamp', 'post_text', 'post_number', 'post_id'] 
    thread_df = pd.DataFrame(columns=thread_cols)
    thread_page_url = thread_url

    while(not lastThreadPage):
        #print(thread_page_url)
        r3 = requests.get(thread_page_url)
        thread_page = r3.text
        thread_page_soup = BeautifulSoup(thread_page, 'html.parser')

        if (thread_page_soup.find('a', text='Next ›') == None):
            lastThreadPage = True
        else:
            thread_page_url = site_url + thread_page_soup.find('a', text='Next ›')['href']

        thread_page_posts = thread_page_soup.find('div', {'id': 'posts'})
        
        try: 
            for post in thread_page_posts.find_all('div', {'class': 'post-wrapper'}):
                userid_url = post.find('a', {'class': 'bigusername'})['href']
                userid = ''.join(filter(lambda x: x.isdigit(), userid_url))

                datetime_raw = post.find('a', {'name': lambda x: x and x.find('post') == 0}).nextSibling.strip()
                date_list = datetime_raw.split(',')[0].split('-')
                iso_date = '-'.join(list(reversed(date_list)))
                hour = int(datetime_raw.split(' ')[1][0:2])
                if(datetime_raw.split(' ')[2] == 'PM' and hour < 12):
                    hour += 12
                hour_str = str(hour)
                if(hour < 10):
                    hour_str = '0' + str(hour)
                minute = datetime_raw.split(':')[1][0:2]
                iso_datetime = iso_date + 'T' + hour_str + ':' + minute

                post_text = ""
                try:
                    post_text = post.find('div', {'class': 'post_message'}).get_text(' ', strip=True)
                except AttributeError as e: 
                    pass

                post_number = int(post.find('a', {'id': lambda x: x and 'postcount' in x, 'target': 'new'}).find('strong').get_text())

                post_id = int(post.find('a', {'id': lambda x: x and 'postcount' in x, 'target': 'new'})['id'].lstrip('postcount'))
                              
                row = pd.DataFrame([[thread_url, userid, iso_datetime, post_text, post_number, post_id]], columns=thread_cols)
                if(len(thread_df)==0):
                    thread_df = row
                else:
                    thread_df = thread_df.append(row, ignore_index=True) #df.append doesn't work inplace
        except:
            row = pd.DataFrame([[thread_url, "", "", "", np.nan, np.nan]], columns=thread_cols) #posts missing, thread may have been deleted
            if(len(thread_df)==0):
                thread_df = row
            else:
                thread_df = thread_df.append(row, ignore_index=True) #df.append doesn't work inplace
    thread_df['post_text'] = thread_df['post_text'].map(lambda x: x.encode('unicode-escape').decode('utf-8'))

    return thread_df

#Save the data into a csv for cleaning
getPosts(thread_url).to_csv("bionix2.csv", encoding='utf-8')