import re
from main_config import *
from filters import *
import random


def random_seed():
    return 0.0


def c_filter(file_to_filter):

    unique_comments = []

    with open(file_to_filter, encoding='utf-8') as f:
        comments = [n.strip() for n in f]

        # random.shuffle(comments, random_seed)

        for comment in comments:

            # split = comment.find(",")

            # if comment[:split].lower() not in bots:
            #     comment = comment[split + 2:]

            #     if 16 <= len(comment.split()) and re.search('[a-zA-Z]', comment):
                if re.search('[a-zA-Z]', comment):
                    for old, new in uncased_regex_replacements:
                        comment = re.sub(old, new, comment, flags=re.I)

                    for old, new in cased_regex_replacements:
                        comment = re.sub(old, new, comment)

                    # UNCASED
                    # x = x.lower()
                    comment = comment.strip()

                    if comment not in unique_comments:
                        unique_comments.append(comment)

    return unique_comments



