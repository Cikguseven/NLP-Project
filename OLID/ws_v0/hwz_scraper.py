from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import requests


def get_posts(target_forum_url):
    site_url = "https://forums.hardwarezone.com.sg"

    thread_cols = ['thread_url', 'username', 'comment'] 
    thread_df = pd.DataFrame(columns=thread_cols)

    response = requests.get(target_forum_url)
    main_forum_page = response.text
    page_soup = BeautifulSoup(main_forum_page, 'lxml')

    regular_threads = page_soup.find('div', class_="structItemContainer-group js-threadList")
    threads = [tag['href'] for tag in regular_threads.find_all("a", {"data-tp-primary": "on"})]

    for thread in tqdm(threads):
        not_last_page = True
        counter = 1

        while not_last_page and counter < 100:
            thread_url = site_url + thread + 'page-' + str(counter)
            thread_response = requests.get(thread_url)
            thread_page = thread_response.text
            thread_page_soup = BeautifulSoup(thread_page, 'lxml')

            counter += 1

            if not thread_page_soup.find('a', class_='pageNav-jump pageNav-jump--next'):
                not_last_page = False

            for post in thread_page_soup.find_all('article', class_="message"):
                results = post.get_text(' ', strip=True)

                if results[1].isspace():
                    results = results[2:]

                username = results.split()[0]

                start_index = re.search(r"((A|P)M|ago|\d+) #\d+(,\d+)*", results).end()
                if start_index + 1 < len(results):
                    comment = results[start_index:]

                    row = pd.DataFrame([[thread_url, username, comment]], columns=thread_cols)
                    thread_df = pd.concat([thread_df, row], ignore_index=True)

    return thread_df

forum = "https://forums.hardwarezone.com.sg/forums/eat-drink-man-woman.16/"

df = get_posts(forum)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def remove_parent(text):
    return re.sub('^(.*said: ).*Click to expand\.\.\.','', text)


def remove_reactions(text):
    return re.sub('Reactions: .*','', text)


df['comment'] = df.comment.apply(remove_parent)
df['comment'] = df.comment.apply(remove_reactions)

df.dropna(inplace = True)

df.to_csv("hwz.csv", encoding='utf-8', index=False)


