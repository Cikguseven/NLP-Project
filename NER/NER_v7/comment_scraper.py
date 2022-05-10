import praw
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../'))
import config


# Function to scrapes comments and authors from a chosen subreddit
def c_scraper(output_file: str, subreddit: str, scrape_limit: int):
    reddit = praw.Reddit(client_id=config.client_id,
                         client_secret=config.client_secret,
                         user_agent=config.user_agent)

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
