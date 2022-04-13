import time 

start = time.time()

# from comment_scraper import c_scraper
from comment_filter import c_filter
from spacy.lang.en.stop_words import STOP_WORDS
import main_config
import conll2003_ner
import re
import skweak
import spacy

spacy.require_gpu()

# Scrape comments from r/sg subreddit.
# c_scraper()

# Filter comments based on length and potential quality/value of content.
# Also cleans up comments to desired format using RegEx by replacing certain
# short forms and lingos with proper english words and abbreviations.
filtered_remaining = c_filter(main_config.remaining_comments)

comment_count = len(filtered_remaining)

train_val_split = main_config.validation_split * comment_count // 100

training_lines = filtered_remaining[train_val_split:]
validation_lines = filtered_remaining[:train_val_split]

comments = [(training_lines, 0), (validation_lines, 1)]

nlp = spacy.load("en_core_web_trf")


# Implementation of combined annotators from conll2003_ner.py module.
full_annotator = conll2003_ner.NERAnnotator().add_all()


# Resolve aggregated labelling functions to create a single annotation for each
# document by estimating a generative model.
unified_model = skweak.aggregation.HMM("hmm", ["LOC", "MISC", "ORG", "PER"], initial_weights={"custom_lf": 25, "core_web_trf": 10, "money_detector": 10, "proper_detector": 0.7})
unified_model.add_underspecified_label("ENT", ["LOC", "MISC", "ORG", "PER"])
unified_model.add_underspecified_label("NOT_ENT", ["O"])


# Create spacy files for training and validation comments
for item in comments:
    spacy_file = main_config.spacy_training if item[1] == 0 else main_config.spacy_validation

    # Import unique filtered comments for testing and validation
    # Run default pipeline from Spacy on them to obtain Doc objects for training

    docs = list(nlp.pipe(item[0]))
    docs = list(full_annotator.pipe(docs))
    docs = unified_model.fit_and_aggregate(docs)

    for doc in docs:
        doc.ents = doc.spans["hmm"]

    skweak.utils.docbin_writer(docs, spacy_file)


end = time.time()
t = round(end - start, 2)

if t > 60:
    duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
else:
    duration = str(t) + 's'

print(f'Time taken: {duration}')