from comment_scraper import c_scraper
from comment_filter import c_filter
import main_config
import gold_labels

import conll2003_ner
import skweak
import spacy


filterrred = c_filter(
    input_file=main_config.hand_labelled_comments,
    shuffle=False,
    remove_username=False,
    length_filter=14,
    uncased=False, 
    unique=True)
    
nlp = spacy.load("en_core_web_trf")
docs = list(nlp.pipe(filterrred))

for i in range(len(filterrred)):
    if gold_labels.answer_key[i]:
        docs[i].set_ents([docs[i].char_span(a[1], a[2], label=a[3]) for a in gold_labels.answer_key[i]])