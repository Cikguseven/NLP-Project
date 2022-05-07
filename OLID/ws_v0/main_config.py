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


def balanced_hwz_getter(undersampled: bool):
    with open('hwz_tagged_1.txt', encoding='utf-8') as f:
        comments = [line.split('|')[0].strip for line in f]

    nn_counter = 0
    oth_counter = 0
    ind_counter = 0
    grp_counter = 0

    nn_comments = []
    ou_comments = []
    ind_comments = []
    grp_comments = []
    oth_comments = []

    for comment in comments:
        if comment[1] == 'NOT':
            if not undersampled or (undersampled and nn_counter < 234):
                nn_comments.append(comment)
                nn_counter += 1
        elif comment[2] == 'UNT':
            ou_comments.append(comment[0])
        elif comment[3] == 'IND':
            if not undersampled or (undersampled and ind_counter < 39):
                ind_comments.append(comment)
                ind_counter += 1
        elif comment[3] == 'GRP':
            if not undersampled or (undersampled and grp_counter < 39):
                grp_comments.append(comment)
                grp_counter += 1
        elif comment[3] == 'OTH':
            if not undersampled or (undersampled and oth_counter < 39):
                oth_comments.append(comment)
                oth_counter += 1

    undersample_comments =  nn_comments + ou_comments + \
        ind_comments + grp_comments + oth_comments

    return undersample_comments
