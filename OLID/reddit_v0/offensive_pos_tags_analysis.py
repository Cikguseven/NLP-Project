import main_config
import spacy
import comment_filter
from os import listdir
from collections import Counter
import tag_freq_analysis

accumulator = Counter()

# custom_models = [f for f in listdir(
#     main_config.NER_model_directory) if 'v' in f]


olid_training_data = main_config.training_tweets_getter()

non_offensive_tweets = []

for tweet in olid_training_data:
    if tweet[2] != 'OFF':
        non_offensive_tweets.append(tweet[1])

print(len(non_offensive_tweets))

filtered_tweets = main_config.preprocess(non_offensive_tweets, False)

tags_dict = {"$": 1,
             "''": 2,
             ",": 3,
             "-LRB-": 4,
             "-RRB-": 5,
             ".": 6,
             ":": 7,
             "ADD": 8,
             "AFX": 9,
             "CC": 10,
             "CD": 11,
             "DT": 12,
             "EX": 13,
             "FW": 14,
             "HYPH": 15,
             "IN": 16,
             "JJ": 17,
             "JJR": 18,
             "JJS": 19,
             "LS": 20,
             "MD": 21,
             "NFP": 22,
             "NN": 23,
             "NNP": 24,
             "NNPS": 25,
             "NNS": 26,
             "PDT": 27,
             "POS": 28,
             "PRP": 29,
             "PRP$": 30,
             "RB": 31,
             "RBR": 32,
             "RBS": 33,
             "RP": 34,
             "SYM": 35,
             "TO": 36,
             "UH": 37,
             "VB": 38,
             "VBD": 39,
             "VBG": 40,
             "VBN": 41,
             "VBP": 42,
             "VBZ": 43,
             "WDT": 44,
             "WP": 45,
             "WP$": 46,
             "WRB": 47,
             "XX": 48,
             "``": 49}

spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')
docs = list(nlp.pipe(filtered_tweets))
all_tags = [[tags_dict[tok.tag_] for tok in doc] for doc in docs]

print(all_tags)

# for pattern in test1.results:
#     length = len(pattern)
#     if length > 5:
#         for l in range(8, length + 1, 3):
#             for start in range(length - l):
#                 accumulator[pattern[start:start+l]] += 1

# print(accumulator.most_common(40))

