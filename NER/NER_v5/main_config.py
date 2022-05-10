from os import getcwd

scraped_comments = "redditsg1.txt"

hand_labelled_comments = '500_cased.txt'

remaining_comments = 'remaining.txt'

model_directory = '../models/'

# total_comments = 500

testing_split = 100
validation_split = 25
training_split = 75

version = "v5_"

all_labels = version + "all_labels.txt"
wrong_labels = version + "wrong_labels.txt"

# testing_count = int(total_comments * 0.15)
# cased_testing_comments = "test50_cased.txt"
# uncased_testing_comments = "test50_uncased.txt"

# validation_count = int(total_comments * 0.15)
validation_str = version + "val_" + str(validation_split) + "_cased"
# validation_comments = validation_str + ".txt"
spacy_validation = validation_str + ".spacy"

# training_count = int(total_comments * 0.7)
training_str = version + "train_" + str(training_split) + "_cased"
# training_comments = training_str + ".txt"
spacy_training = training_str + ".spacy"

# with open(hand_labelled_comments) as f:
#     testing_lines = [next(f).strip() for x in range(testing_count)]
#     validation_lines = [next(f).strip() for x in range(validation_count)]
#     training_lines = [next(f).strip() for x in range(training_count)]
#     all_lines = [*testing_lines, *validation_lines, *training_lines]

with open(hand_labelled_comments) as f:
    testing_lines = [line.strip() for line in f]

if __name__ == "__main__":
    import gold_labels
    import re
    import main_config
    from comment_filter import *

    # for i in range(500):
    #     linecopy = main_config.testing_lines[i]
    #     temp = []
    #     for label in gold_labels.answer_key[i]:
    #         index = re.search(r'(?<![^\W_])' + label[0] + r'(?![^\W_])', linecopy).start()
    #         end = index + len(label[0])
    #         if label[0] != linecopy[label[1]:label[2]]:
    #             print(i)
    #             print([label[0], index, end, label[3]])
    #             print()
    #         linecopy = re.sub(r'(?<![^\W_])' + label[0] + r'(?![^\W_])', ' ' * len(label[0]), linecopy, 1)

    for x in c_filter(hand_labelled_comments):
        print(x)



