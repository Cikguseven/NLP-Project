from collections import Counter
import emoji
import filters
import re
import wordsegment

olid_directory = '../OLID_dataset/'
model_directory = '../models/'

training_tweet_file = olid_directory + 'olid-training-v1.tsv'

test_tweets_a_file = olid_directory + 'testset-levela.tsv'
test_tweets_b_file = olid_directory + 'testset-levelb.tsv'
test_tweets_c_file = olid_directory + 'testset-levelc.tsv'
test_tweet_files = (test_tweets_a_file, test_tweets_b_file, test_tweets_c_file)

test_answers_a_file = olid_directory + 'labels-levela.csv'
test_answers_b_file = olid_directory + 'labels-levelb.csv'
test_answers_c_file = olid_directory + 'labels-levelc.csv'
test_answer_files = (test_answers_a_file,
                     test_answers_b_file, test_answers_c_file)

validation_split = 25

version = 'v1_cased_'

output_training = version + 'training.spacy'
output_vaildation = version + 'vaildation.spacy'


def training_tweets_getter():
    with open(training_tweet_file, encoding='utf-8') as f:
        tweets = [line.split('\t') for line in f]

    tweets.pop(0)
    for tweet in tweets:
        tweet[4] = tweet[4].strip()

    return tweets


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
        for old, new in filters.uncased_regex_replacements:
            tweets[i] = re.sub(old, new, tweets[i], flags=re.I)

        for old, new in filters.cased_regex_replacements:
            tweets[i] = re.sub(old, new, tweets[i])

        # Uncased text
        if uncased:
            tweets[i] = tweets[i].lower()

    return tweets


def k_most_frequent_words(data: list, k: int):
    split_data = [word for sentence in data for word in sentence.split()]
    frequency = Counter(split_data)
    return frequency.most_common(k)
