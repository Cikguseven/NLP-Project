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
        thread_url = site_url + thread + 'page-'

        not_last_page = True
        counter = 1

        while not_last_page and counter < 100:
            thread_response = requests.get(thread_url + str(counter))
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

                start_index = re.search(r"((A|P)M|ago|\d+) #[0-9,]", results).end()
                if start_index + 1 < len(results):
                    comment = results[start_index:]

                    row = pd.DataFrame([[thread, username, comment]], columns=thread_cols)
                    thread_df = pd.concat([thread_df, row], ignore_index=True)

    return thread_df

forum = "https://forums.hardwarezone.com.sg/forums/eat-drink-man-woman.16/"

get_posts(forum).to_csv("hwz.csv", encoding='utf-8')
