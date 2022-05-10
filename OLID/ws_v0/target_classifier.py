from bisect import bisect_left
from tqdm import tqdm
import labelling_functions_data
import main_config
import re
import spacy
import statistics


def take_closest(input_list, input_number):
    pos = bisect_left(input_list, input_number)
    if pos == 0:
        return input_list[0]
    if pos == len(input_list):
        return input_list[-1]
    before = input_list[pos - 1]
    after = input_list[pos]
    if after - input_number < input_number - before:
        return after
    else:
        return before


def entity_to_target(entity: str):
    uncased_entity = entity.lower()

    if entity in ('LOC', 'MISC', 'ORG', 'PER'):
        if entity == 'PER':
            return 'IND'
        else:
            return 'OTH'

    elif uncased_entity in labelling_functions_data.individual:
        return 'IND'

    elif uncased_entity in labelling_functions_data.group:
        return 'GRP'

    elif uncased_entity in labelling_functions_data.other:
        return 'OTH'


def weak_classifier(sentences: list):
    valid_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    results = []

    spacy.require_gpu()

    nlp = spacy.load("en_core_web_trf")
    nlp_ner = spacy.load(main_config.NER_model)

    docs = list(nlp.pipe(sentences))
    docs_ner = list(nlp_ner.pipe(sentences))

    for i in tqdm(range(len(docs))):

        total_targets = []

        offensive_word_pos = [result.start() for key in labelling_functions_data.offensive_lexicon if (result := re.search(r'(?<![^\W_])' + key + r'(?![^\W_])', docs[i].text.lower()))]
        noun_verb_pos = {token.idx: token.text for token in docs[i] if token.tag_ in valid_tags}
        ent_pos = {x.start_char: x.label_ for x in docs_ner[i].ents}

        if noun_verb_pos and offensive_word_pos:
            average_offensive_pos = statistics.mean(offensive_word_pos)
            noun_verb_pos_list = [x for x in noun_verb_pos if x != average_offensive_pos]
            if len(noun_verb_pos_list) > 0:
                closest_index = take_closest(noun_verb_pos_list, average_offensive_pos)
                if closest_index in ent_pos:
                    total_targets.append(entity_to_target(ent_pos[closest_index]))
                else:
                    total_targets.append(entity_to_target(noun_verb_pos[closest_index]))

        total_targets += [result for word in list(noun_verb_pos.values()) + list(ent_pos.values()) if (result := entity_to_target(word))]
        new_targets = list(filter(None, total_targets))

        if new_targets:
            results.append(max(new_targets, key=new_targets.count))
        else:
            results.append('OTH')

    return results
