import praw
import random
from main_config import *
import time


def random_seed():
    return 0.0


# Function to scrapes comments and authors from r/sg subreddit
def c_scraper():
    start = time.time()

    reddit = praw.Reddit(client_id='KQpseqhxhSGWZ0CXQrdCkQ',
                         client_secret='Xsbo93sniLTYdisbEqlIoBzTNpZDFA',
                         user_agent='MyBot/0.0.1')

    subreddit = reddit.subreddit('Singapore')

    time_filters = ['day', 'week', 'month', 'year', 'all']

    viewed_posts = []

    counter = 0

    with open(scraped_comments, "w", encoding='utf-8') as f:
        for filter in time_filters:
            for post in subreddit.top(filter, limit=30):
                if post.id not in viewed_posts:
                    print(f'{filter}|{post.id}')
                    viewed_posts.append(post.id)
                    submission = reddit.submission(id=post.id)
                    submission.comments.replace_more(limit=0)

                    for comment in submission.comments.list():
                        x = comment.body.replace("\n", " ")
                        f.write(f"{post.id}|{comment.author}, {x}\n")
                        counter += 1

        lines = f.readlines()
    
    # Shuffles comments based on fixed seed
    random.shuffle(lines, random_seed)
    
    with open(scraped_comments, "w", encoding='utf-8') as f:
        f.writelines(lines)

    end = time.time()
    duration = round(end - start, 2)

    if duration > 60:
        duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
    else:
        duration = str(duration) + 's'
