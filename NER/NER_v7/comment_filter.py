import os
import sys
import random
import re

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import shared_filters


def random_seed():
    return 0.0


# Function to filter reddit comments using RegEx and returns list of filtered comments
# Arguments:
# input_file - Absolute/relative filepath of .txt file with reddit comments
# shuffle - Randomly shuffles comments based on fixed seed
# remove_username - Removes username in line of text based on first comma
# length_filter - Comments must contain at least chosen number of words
# uncased - Converts comments to lowercase
# unique - Only returns unique comments
def c_filter(
    input_file: str,
    shuffle: bool,
    remove_username: bool,
    length_min: int,
    length_max: int,
    uncased: bool, 
    unique: bool):
    
    with open(input_file, encoding='utf-8') as f:
        comments = [n.strip() for n in f]
        output_comments = []

        if shuffle:
            random.shuffle(comments, random_seed)

        for comment in comments:

            if remove_username:
                split = comment.find(",")

                if comment[:split].lower() not in shared_filters.bots:
                    comment = comment[split + 2:]

            if length_min <= len(comment.split()) <= length_max and re.search('[a-zA-Z]', comment):
                for old, new in shared_filters.uncased_regex_replacements:
                    comment = re.sub(old, new, comment, flags=re.I)

                for old, new in shared_filters.cased_regex_replacements:
                    comment = re.sub(old, new, comment)

                if uncased:
                    comment = comment.lower()

                comment = comment.strip()

                if unique:
                    if comment not in output_comments:
                        output_comments.append(comment)
                else:
                    output_comments.append(comment)

    print('Comments filtered')
    return output_comments
