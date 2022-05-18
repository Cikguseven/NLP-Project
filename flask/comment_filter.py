import os
import sys
import re
import emoji
import wordsegment

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import shared_filters


def random_seed():
    return 0.1


# Function to filter comments based on input file/list
# Arguments:
# uncased - Converts comments to lowercase
# edmw - Translates EDMW lingo to English
def c_filter(
    uncased: bool, 
    edmw=False,
    **kwargs):

    output_comments = []

    if kwargs.get('input_file', None):
        with open(kwargs['input_file'], encoding='utf-8') as f:
            comments = [n.strip() for n in f]
    else:
        comments = kwargs['input_list']        

    for comment in comments:
        
        if re.search('[a-zA-Z]', comment):

            # User mention replacement
            if comment.find('@USER') != comment.rfind('@USER'):
                comment = comment.replace('@USER', '')
                comment = '@USERS ' + comment

            # Hashtag segmentation
            if '#' in comment:
                wordsegment.load()
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

            if uncased:
                comment = comment.lower()

            comment = comment.strip()

            output_comments.append(comment)
   
    return output_comments
