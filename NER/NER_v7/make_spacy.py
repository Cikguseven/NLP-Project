from comment_scraper import c_scraper
from comment_filter import c_filter
import main_config
import gold_labels

import modified_conll2003_ner
import skweak
import spacy

spacy.require_gpu()

def spacy_file_creator(
    comments: list,
    load_spacy_model: str,
    weak_supervision_mode: bool,
    training_validation_split: int,
    output_training: str,
    output_vaildation: str):

    # Run spaCy's transformer pipeline on comments to obtain Doc objects and labels
    nlp = spacy.load(load_spacy_model)
    docs = list(nlp.pipe(comments))


    # Adding labels from combined annotators in conll2003_ner.py module
    full_annotator = modified_conll2003_ner.NERAnnotator().add_all()
    docs = list(full_annotator.pipe(docs))


    # Resolve aggregated labelling functions to create a single annotation for each
    # document by estimating a generative model.
    if weak_supervision_mode:

        unified_model = skweak.aggregation.HMM("hmm", ["LOC", "MISC", "ORG", "PER"], initial_weights={"custom_lf": 60, "core_web_trf": 10, "money_detector": 10, "proper_detector": 0.7})
        unified_model.add_underspecified_label("NOT_ENT", ["O"])

    else:
        # Adding gold standard hand labels to comments if weak supervision is not used
        for i in range(len(comments)):
            if gold_labels.answer_key[i]:
                docs[i].set_ents([docs[i].char_span(a[1], a[2], label=a[3]) for a in gold_labels.answer_key[i]])

        unified_model = skweak.aggregation.HMM("hmm", ["LOC", "MISC", "ORG", "PER"], initial_weights={"custom_lf": 0})

    unified_model.add_underspecified_label("ENT", ["LOC", "MISC", "ORG", "PER"])

    docs = unified_model.fit_and_aggregate(docs)

    for doc in docs:
        doc.ents = doc.spans["hmm"]

    # Serialize Doc objects as spaCy files for training and validation comments
    split = training_validation_split * len(docs) // 100
    skweak.utils.docbin_writer(docs[split:], output_training)
    skweak.utils.docbin_writer(docs[:split], output_vaildation)


if __name__ == '__main__':
    import time 
    start = time.time()
    # # Scrape comments from r/sg subreddit.
    # c_scraper(
    # output_file=main_config.scraped_comments,
    # subreddit='Singapore',
    # scrape_limit=30)

    # Import unique filtered comments for testing and validation
    filtered_comments = c_filter(
        input_file=main_config.hand_labelled_comments,
        shuffle=False,
        remove_username=False,
        length_min=14,
        length_max=99999,
        uncased=False,
        unique=True)

    spacy_file_creator(
        comments=filtered_comments[:],
        load_spacy_model="en_core_web_trf",
        weak_supervision_mode=True,
        training_validation_split=main_config.validation_split,
        output_training=main_config.spacy_validation_file,
        output_vaildation=main_config.spacy_training_file)

    end = time.time()
    t = round(end - start, 2)

    if t > 60:
        duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
    else:
        duration = str(t) + 's'

    print(f'Time taken: {duration}')
