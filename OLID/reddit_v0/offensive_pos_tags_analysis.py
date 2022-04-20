import main_config
import spacy
import comment_filter
from os import listdir
from collections import Counter

accumulator_non = Counter()
accumulator_off = Counter()

tags_dict = {"$": 10,
             "''": 11,
             ",": 12,
             "-LRB-": 13,
             "-RRB-": 14,
             ".": 15,
             ":": 16,
             "ADD": 17,
             "AFX": 18,
             "CC": 19,
             "CD": 20,
             "DT": 21,
             "EX": 22,
             "FW": 23,
             "HYPH": 24,
             "IN": 25,
             "JJ": 26,
             "JJR": 27,
             "JJS": 28,
             "LS": 29,
             "MD": 30,
             "NFP": 31,
             "NN": 32,
             "NNP": 33,
             "NNPS": 34,
             "NNS": 35,
             "PDT": 36,
             "POS": 37,
             "PRP": 38,
             "PRP$": 39,
             "RB": 40,
             "RBR": 41,
             "RBS": 42,
             "RP": 43,
             "SYM": 44,
             "TO": 45,
             "UH": 46,
             "VB": 47,
             "VBD": 48,
             "VBG": 49,
             "VBN": 50,
             "VBP": 51,
             "VBZ": 52,
             "WDT": 53,
             "WP": 54,
             "WP$": 55,
             "WRB": 56,
             "XX": 57,
             "``": 58}

non_offensive_tweets = []
non_values = []
offensive_tweets = []
off_values = []

data = [(non_offensive_tweets, 0, non_values), (offensive_tweets, 1, off_values)]

olid_training_data = main_config.training_tweets_getter()

for tweet in olid_training_data:
    if tweet[2] != 'OFF':
        non_offensive_tweets.append(tweet[1])
    else:
        offensive_tweets.append(tweet[1])

non_offensive_tweets = main_config.preprocess(non_offensive_tweets, False)
offensive_tweets = main_config.preprocess(offensive_tweets, False)

spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')

for sets in data:
    docs = list(nlp.pipe(sets[0]))
    all_tags = [[tags_dict[tok.tag_] for tok in doc] for doc in docs]
    for pattern in all_tags:
        length = len(pattern)
        if length > 5:
            for l in range(4, length + 1):
                for start in range(length - l):
                    if sets[1] == 0:
                        accumulator_non[tuple(pattern[start:start+l])] += 1
                    else:
                        accumulator_off[tuple(pattern[start:start+l])] += 1

non_values = [[get_key(y) for y in x[0]] for x in accumulator_non.most_common(80)]
off_values = [[get_key(y) for y in x[0]] for x in accumulator_off.most_common(80)]

print(non_values)
print(off_values)


def get_key(val):
    for key, value in tags_dict.items():
         if val == value:
             return key

    return "key doesn't exist"



