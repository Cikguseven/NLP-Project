import time
start = time.time()

from spacy.lang.en.stop_words import STOP_WORDS
import main_config
import gold_labels
import spacy
import re
from os import listdir

spacy.require_gpu()


def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 0.25


label_dict = {
    'CARDINAL': None,
    'DATE': None,
    'EVENT': 'MISC',
    'FAC': 'MISC',
    'GPE': 'LOC',
    'LANGUAGE': 'MISC',
    'LAW': 'MISC',
    'LOC': 'LOC',
    'MISC': 'MISC',
    'MONEY': None,
    'NORP': 'MISC',
    'ORDINAL': None,
    'ORG': 'ORG',
    'PERCENT': None,
    'PER': 'PER',
    'PERSON': 'PER',
    'PRODUCT': 'MISC',
    'QUANTITY': None,
    'TIME': None,
    'WORK_OF_ART': 'MISC'
}


def evaluate_model(models: list, answer_key: list, testing_lines: list, all_labels: str, wrong_labels: str):

    for model in models:

        if model[0] == 'e':
            nlp = spacy.load(model)
        else:
            nlp = spacy.load(main_config.model_directory + model + '/model-best')

        docs = list(nlp.pipe(testing_lines))

        stopwords = nlp.Defaults.stop_words
        stopwords.add('to be')
        stopwords.add('think it')

        tp = 0
        fp = 0

        new_all_labels = []

        with open(all_labels, 'w') as all_l, open(wrong_labels, 'w') as wrong_l:
            for i in range(len(testing_lines)):
                testing_lines_copy = testing_lines[i]
                sentence_labels = []

                for X in docs[i].ents:

                    # Filter stopwords
                    if X.text.lower() not in stopwords and label_dict[X.label_]:

                        new_label = [X.text, X.start_char, X.end_char, label_dict[X.label_]]
                        
                        sentence_labels.append(new_label)
                        all_l.write(f'{", ".join(map(str, new_label))}\n')

                        if new_label in answer_key[i]:
                            tp += 1
                        else:
                            fp += 1
                            wrong_l.write(f'{", ".join(map(str, new_label))}\n')

                new_all_labels.append(sentence_labels)


        precision = tp / (tp + fp)
        recall = tp / recursive_len(answer_key)
        f1 = 2 * precision * recall / (precision + recall)

        print(f'{model}, {precision}, {recall}, {f1}')
        # print(new_all_labels)


    end = time.time()
    t = round(end - start, 2)

    if t > 60:
        duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
    else:
        duration = str(t) + 's'

    print(f'Time taken: {duration}')


specific_model = []

custom_models = [f for f in listdir(main_config.model_directory) if 'v5' in f]

spacy_models = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_core_web_trf']

all_models = custom_models + spacy_models

evaluate_model(custom_models, gold_labels.answer_key, main_config.testing_lines, main_config.all_labels, main_config.wrong_labels)
