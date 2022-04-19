scraped_comments = "redditsg.txt"

hand_labelled_comments = '500_cased.txt'

remaining_comments = 'remaining.txt'

model_directory = '../models/'
NER_model_directory = '../../NER/models/'

validation_split = 25
training_split = 100 - validation_split

version = "v7_"

all_labels = version + "all_labels.txt"
wrong_labels = version + "wrong_labels.txt"

validation_str = version + "val_" + str(validation_split) + "_cased"
spacy_validation_file = validation_str + ".spacy"

training_str = version + "train_" + str(training_split) + "_cased"
spacy_training_file = training_str + ".spacy"

# with open(hand_labelled_comments) as f:
#     testing_lines = [next(f).strip() for x in range(testing_count)]
#     validation_lines = [next(f).strip() for x in range(validation_count)]
#     training_lines = [next(f).strip() for x in range(training_count)]
#     all_lines = [*testing_lines, *validation_lines, *training_lines]

with open(hand_labelled_comments) as f:
    testing_lines = [line.strip() for line in f]

with open(remaining_comments) as f:
    remaining_lines = [line.strip() for line in f]


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


if __name__ == "__main__":
    import gold_labels
    import re
    import main_config
    from comment_filter import *

    for i in range(500):
        linecopy = main_config.testing_lines[i]
        temp = []
        for label in gold_labels.answer_key[i]:
            index = re.search(r'(?<![^\W_])' + label[0] + r'(?![^\W_])', linecopy).start()
            end = index + len(label[0])
            if label[0] != linecopy[label[1]:label[2]]:
                print(i + 1)
                print([label[0], index, end, label[3]])
                print()
            linecopy = re.sub(r'(?<![^\W_])' + label[0] + r'(?![^\W_])', ' ' * len(label[0]), linecopy, 1)

    # for x in c_filter(hand_labelled_comments):
    #     print(x)
