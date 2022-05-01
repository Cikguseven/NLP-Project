from collections import Counter
import comment_filter
import emoji
import os
import re
import sys
import wordsegment

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import shared_filters

olid_directory = '../OLID_dataset/'
model_directory = '../models/'
NER_model = '../../NER/models/v7/model-best'

training_tweet_file = olid_directory + 'olid-training-v1.tsv'
balanced_tweet_file = olid_directory + 'olid-combined.tsv'

test_tweets_a_file = olid_directory + 'testset-levela.tsv'
test_tweets_b_file = olid_directory + 'testset-levelb.tsv'
test_tweets_c_file = olid_directory + 'testset-levelc.tsv'
test_tweet_files = (test_tweets_a_file, test_tweets_b_file, test_tweets_c_file)

test_answers_a_file = olid_directory + 'labels-levela.csv'
test_answers_b_file = olid_directory + 'labels-levelb.csv'
test_answers_c_file = olid_directory + 'labels-levelc.csv'
test_answer_files = (test_answers_a_file,
                     test_answers_b_file, test_answers_c_file)

scraped_comments = 'redditsg.txt'
hand_labelled_comments = 'redditsg_testing.txt'
hand_labelled_comments_csv = 'redditsg_testing.csv'
hand_labelled_comments_uncased_csv = 'redditsg_testing_uncased.csv'
remaining_comments = 'redditsg_trainval.txt'

gold_labels = 'gold_labels_reddit.txt'
with open(gold_labels) as f:
    answers = [line.strip().split() for line in f]

validation_split = 0.25

version = 'ws_v1_60a_15b_lexicon1_'

spacy_training_file = version + 'training.spacy'
spacy_validation_file = version + 'vaildation.spacy'


def training_tweets_getter(unlabelled: bool):
    with open(training_tweet_file, encoding='utf-8') as f:
        tweets = [line.split('\t') for line in f]

    tweets.pop(0)
    for tweet in tweets:
        tweet[4] = tweet[4].strip()

    if unlabelled:        
        output_tweets = [tweet[1] for tweet in tweets]
        return output_tweets
    else:
        return tweets


def balanced_tweets_getter(analysis_set: bool):
    with open(balanced_tweet_file, encoding='utf-8') as f:
        tweets = [line.split('\t') for line in f]

    tweets.pop(0)

    if analysis_set:
        for tweet in tweets:
            tweet[4] = tweet[4].strip()

        nn_tweets = []
        nn_counter = 0

        ou_tweets = []

        ot_tweets = []
        ot_counter = 0

        ind_tweets = []
        ind_counter = 0

        grp_tweets = []
        grp_counter = 0

        oth_tweets = []

        for tweet in tweets:
            if nn_counter < 1102 and tweet[2] == 'NOT':
                nn_tweets.append(tweet[1])
                nn_counter += 1
            elif tweet[3] == 'UNT':
                ou_tweets.append(tweet[1])
            elif ot_counter < 551 and tweet[3] == 'TIN':
                ot_tweets.append(tweet[1])
                ot_counter += 1

            if tweet[4] == 'OTH':
                oth_tweets.append(tweet[1])
            elif tweet[4] == 'IND' and ind_counter < 430:
                ind_tweets.append(tweet[1])
                ind_counter += 1
            elif tweet[4] == 'GRP' and grp_counter < 430:
                grp_tweets.append(tweet[1])
                grp_counter += 1

        output_tweets = nn_tweets + ou_tweets + \
            ot_tweets + oth_tweets + ind_tweets + grp_tweets

        return output_tweets

    else:
        output_tweets = [tweet[1] for tweet in tweets]
        return output_tweets


def test_tweets_getter():
    tweets = []

    for file in test_tweet_files:
        with open(file, encoding='utf-8') as f:
            tweets += [line.split('\t')[1]
                       for line in f if line.split('\t')[1] != 'tweet\n']

    return tweets


def answers_getter():
    answers = []

    for file in test_answer_files:
        with open(file) as f:
            answers += [line.split(',')[1].strip() for line in f]

    return answers


def answers_frequency(answers: list):
    frequency = []

    for label in answers:
        if label[0] == 'NOT':
            frequency.append('NOT')
        else:
            frequency.append('OFF')

            if label[1] == 'UNT':
                frequency.append('UNT')
            else:
                frequency.append('TIN')

                if label[2] == 'IND':
                    frequency.append('IND')
                elif label[2] == 'GRP':
                    frequency.append('GRP')
                else:
                    frequency.append('OTH')

    return frequency


def preprocess(tweets: list, uncased: bool):
    wordsegment.load()

    for i in range(len(tweets)):

        # User mention replacement
        if tweets[i].find('@USER') != tweets[i].rfind('@USER'):
            tweets[i] = tweets[i].replace('@USER', '')
            tweets[i] = '@USERS ' + tweets[i]

        # Hashtag segmentation
        line_tokens = tweets[i].split(' ')
        for j, t in enumerate(line_tokens):
            if t.find('#') == 0:
                line_tokens[j] = ' '.join(wordsegment.segment(t))
        tweets[i] = ' '.join(line_tokens)

        # Emoji to word
        tweets[i] = emoji.demojize(tweets[i])

        # Formatting and slang replacement
        for old, new in shared_filters.uncased_regex_replacements:
            tweets[i] = re.sub(old, new, tweets[i], flags=re.I)

        for old, new in shared_filters.cased_regex_replacements:
            tweets[i] = re.sub(old, new, tweets[i])

        # Uncased text
        if uncased:
            tweets[i] = tweets[i].lower()

        tweets[i] = tweets[i].strip()

    return tweets


def k_most_frequent_words(data: list, k: int):
    split_data = [word for sentence in data for word in sentence.split()]
    frequency = Counter(split_data)
    return frequency.most_common(k)
