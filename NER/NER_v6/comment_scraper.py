import main_config
import praw
import random
import time

sys.path.insert(1, os.path.join(sys.path[0], '../'))
import config


# Function to scrapes comments and authors from r/sg subreddit
def c_scraper():
    start = time.time()

    reddit = praw.Reddit(client_id=config.client_id,
                         client_secret=config.client_secret,
                         user_agent=config.user_agent)


# Function to scrapes comments and authors from a chosen subreddit
def c_scraper(output_file: str, subreddit: str, scrape_limit: int):
    start = time.time()

    reddit = praw.Reddit(client_id='KQpseqhxhSGWZ0CXQrdCkQ',
                         client_secret='Xsbo93sniLTYdisbEqlIoBzTNpZDFA',
                         user_agent='MyBot/0.0.1')

    target_subreddit = reddit.subreddit(subreddit)

    time_filters = ['day', 'week', 'month', 'year', 'all']

    # Avoid duplicate posts
    viewed_posts = []

    counter = 0

    with open(output_file, "w", encoding='utf-8') as f:
        for filter in time_filters:
            for post in target_subreddit.top(filter, limit=scrape_limit):
                if post.id not in viewed_posts:
                    print(f'{filter}|{post.id}')
                    viewed_posts.append(post.id)
                    submission = reddit.submission(id=post.id)
                    submission.comments.replace_more(limit=0)

                    for comment in submission.comments.list():
                        new_comment = comment.body.replace("\n", " ")
                        f.write(f"{post.id}|{comment.author}, {new_comment}\n")
                        counter += 1

    print(f'Comments scraped: {counter}')

    end = time.time()
    duration = round(end - start, 2)

    if duration > 60:
        duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
    else:
        duration = str(duration) + 's'

    print(f'Time taken: {duration}')
