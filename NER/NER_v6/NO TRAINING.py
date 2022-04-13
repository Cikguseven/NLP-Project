import time 

start = time.time()

# from comment_scraper import c_scraper
from comment_filter import c_filter
from spacy.lang.en.stop_words import STOP_WORDS
import main_config
from gold_labels import answer_key
import conll2003_ner
import re
import skweak
import spacy


# Scrape comments from r/sg subreddit.
# c_scraper()

# Filter comments based on length and potential quality/value of content.
# Also cleans up comments to desired format using RegEx by replacing certain
# short forms and lingos with proper english words and abbreviations.
filtered_testing = c_filter(main_config.hand_labelled_comments)

spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")


# Implementation of combined annotators from conll2003_ner.py module.
full_annotator = conll2003_ner.NERAnnotator().add_all()


# Resolve aggregated labelling functions to create a single annotation for each
# document by estimating a generative model.
unified_model = skweak.aggregation.HMM("hmm", ["LOC", "MISC", "ORG", "PER"], initial_weights={"custom_lf": 60, "core_web_trf": 10, "money_detector": 10, "proper_detector": 0.7})
unified_model.add_underspecified_label("ENT", ["LOC", "MISC", "ORG", "PER"])
unified_model.add_underspecified_label("NOT_ENT", ["O"])


# Import unique filtered comments for testing and validation
# Run default pipeline from Spacy on them to obtain Doc objects for training

docs = list(nlp.pipe(filtered_testing))
docs = list(full_annotator.pipe(docs))
docs = unified_model.fit_and_aggregate(docs)

for doc in docs:
    doc.ents = doc.spans["hmm"]

stopwords = nlp.Defaults.stop_words
stopwords.add('to be')
stopwords.add('think it')

tp = 0
fp = 0

new_all_labels = []

with open("no_training_all_labels.txt", 'w') as all_l, open("no_training_wrong_labels.txt", 'w') as wrong_l:
    for i in range(len(filtered_testing)):
        testing_lines_copy = filtered_testing[i]
        sentence_labels = []

        for X in docs[i].ents:

            # Filter stopwords
            if X.text.lower() not in stopwords and main_config.label_dict[X.label_]:

                new_label = [X.text, X.start_char, X.end_char, label_dict[X.label_]]
                
                sentence_labels.append(new_label)
                all_l.write(f'{", ".join(map(str, new_label))}, {i + 1}\n')

                if new_label in answer_key[i]:
                    tp += 1
                else:
                    fp += 1
                    wrong_l.write(f'{", ".join(map(str, new_label))}, {i + 1}\n')

        new_all_labels.append(sentence_labels)


precision = tp / (tp + fp)
recall = tp / main_config.recursive_len(answer_key)
f1 = 2 * precision * recall / (precision + recall)

print(f'{precision}, {recall}, {f1}')

end = time.time()
t = round(end - start, 2)

if t > 60:
    duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
else:
    duration = str(t) + 's'

print(f'Time taken: {duration}')
