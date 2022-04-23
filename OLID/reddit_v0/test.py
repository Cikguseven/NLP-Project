from bad_words import offensive_lexicon
from bisect import bisect_left
from collections import Counter 
import labelling_functions_data
import main_config
import re
import spacy
import statistics
import comment_filter


def take_closest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def entity_to_target(entity: str):
    if entity == 'PER' or entity.lower() in labelling_functions_data.individual:
        return 'IND'

    elif entity == 'MISC' or entity == 'LOC' or entity.lower() in labelling_functions_data.group:
        return 'GRP'

    return 'OTH'


def target_classifier(sentences: list):
    valid_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']

    results = []

    spacy.require_gpu()

    nlp = spacy.load("en_core_web_trf")
    nlp_ner = spacy.load(main_config.NER_model)

    docs = list(nlp.pipe(sentences))
    docs_ner = list(nlp_ner.pipe(sentences))

    for i in range(len(docs)):
        offensive_word_pos = [result.start() for key in offensive_lexicon if (result := re.search(r'(?<![^\W_])' + key + r'(?![^\W_])', docs[i].text.lower()))]
        noun_pos = {token.idx: token.text for token in docs[i] if token.tag_ in valid_tags}
        ent_pos = {x.start_char: x.label_ for x in docs_ner[i].ents}

        if noun_pos and offensive_word_pos:
            closest_index = take_closest([x for x in noun_pos], statistics.mean(offensive_word_pos))
            if closest_index in ent_pos:
                print('ent_pos')
                prediction = entity_to_target(ent_pos[closest_index])
            else:
                print('noun_pos')
                prediction = entity_to_target(noun_pos[closest_index])

        else:
            all_targets = [entity_to_target(word) for word in list(noun_pos.values()) + list(ent_pos.values())]
            print(all_targets)
            if all_targets:
                print('most_common')
                prediction = Counter(all_targets).most_common(1)[0][0]
            else:
                print('not any')
                prediction = 'IND'

        results.append(prediction)
        print(f'{docs[i].text} | {prediction}')
        print()

    return results

if __name__ == '__main__':
    olid_balanced_tweets = main_config.balanced_tweets_getter()

    # Import unique filtered comments for testing
    filtered_tweets = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=olid_balanced_tweets)

    target_classifier(filtered_tweets[3480:])
