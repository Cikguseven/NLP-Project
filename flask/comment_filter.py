import os
import sys
import random
import re
import emoji
import wordsegment

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import shared_filters


def random_seed():
    return 0.1


# Function to filter reddit comments using RegEx and returns list of filtered comments
# Arguments:
# input_file - Absolute/relative filepath of .txt file with reddit comments
# shuffle - Randomly shuffles comments based on fixed seed
# remove_username - Removes username in line of text based on first comma
# length_min - Comments must contain at least chosen number of words
# length_max - Comments must contain at most chosen number of words
# uncased - Converts comments to lowercase
# unique - Only returns unique comments
def c_filter(
    shuffle: bool,
    remove_username: bool,
    remove_commas: bool,
    length_min: int,
    length_max: int,
    uncased: bool, 
    unique: bool,
    edmw=False,
    **kwargs):

    wordsegment.load()

    output_comments = []

    if kwargs.get('input_file', None):
        with open(kwargs['input_file'], encoding='utf-8') as f:
            comments = [n.strip() for n in f]
    else:
        comments = kwargs['input_list']        

    if shuffle:
        random.shuffle(comments, random_seed)

    for comment in comments:

        if remove_username:
            split = comment.find(",")

            if comment[:split].lower() not in shared_filters.bots:
                comment = comment[split + 2:]

        if length_min <= len(comment.split()) <= length_max and re.search('[a-zA-Z]', comment):

            # User mention replacement
            if comment.find('@USER') != comment.rfind('@USER'):
                comment = comment.replace('@USER', '')
                comment = '@USERS ' + comment

            # Hashtag segmentation
            line_tokens = comment.split(' ')
            for j, t in enumerate(line_tokens):
                if t.find('#') == 0:
                    line_tokens[j] = ' '.join(wordsegment.segment(t))
            comment = ' '.join(line_tokens)

            comment = emoji.demojize(comment)

            if edmw:
                for old, new in shared_filters.edmw_replacements:
                    comment = re.sub(old, new, comment, flags=re.I)

            for old, new in shared_filters.uncased_regex_replacements:
                comment = re.sub(old, new, comment, flags=re.I)

            for old, new in shared_filters.cased_regex_replacements:
                comment = re.sub(old, new, comment)

            if remove_commas:
                comment = comment.replace(',', '')

            if uncased:
                comment = comment.lower()

            comment = comment.strip()

            if unique:
                if comment not in output_comments:
                    output_comments.append(comment)
            else:
                output_comments.append(comment)

    print(f'{len(output_comments)} filtered comments')
    
    return output_comments
