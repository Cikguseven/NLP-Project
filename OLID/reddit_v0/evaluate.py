from collections import Counter
from os import listdir
import main_config
import comment_filter
import spacy
import time

# spacy.require_gpu()


def evaluate_model(
        models: list,
        test_comments: list,
        test_answers: list,
        offensive_threshold: int,
        targeted_threshold: int):

    category_frequency = Counter(test_answers)

    print(category_frequency)

    uncased_test_comments = [comment.lower() for comment in test_comments]

    for model in models:
        start = time.time()

        print(model)

        nlp = spacy.load(main_config.model_directory + model + '/model-best')

        if 'uncased' in model:
            docs = list(nlp.pipe(uncased_test_comments))
        else:
            docs = list(nlp.pipe(test_comments))

        true_positive_not = 0
        false_positive_not = 0

        true_positive_off = 0
        false_positive_off = 0

        true_positive_unt = 0
        false_positive_unt = 0

        true_positive_tin = 0
        false_positive_tin = 0

        true_positive_ind = 0
        false_positive_ind = 0

        true_positive_grp = 0
        false_positive_grp = 0

        true_positive_oth = 0
        false_positive_oth = 0

        for i in range(len(test_comments)):
            result = docs[i].cats

            if test_answers[i] == 'OFF':
                if result["offensive"] > offensive_threshold:
                    true_positive_off += 1
                else:
                    false_positive_not += 1

            elif test_answers[i] == 'NOT':
                if result["offensive"] < offensive_threshold:
                    true_positive_not += 1
                else:
                    false_positive_off += 1

            elif test_answers[i] == 'TIN':
                if result["targeted"] > targeted_threshold:
                    true_positive_tin += 1
                else:
                    false_positive_unt += 1

            elif test_answers[i] == 'UNT':
                if result["targeted"] < targeted_threshold:
                    true_positive_unt += 1
                else:
                    false_positive_tin += 1

            else:
                result.pop('offensive')
                result.pop('targeted')
                prediction = max(result, key=result.get)

                if prediction == 'individual':
                    if test_answers[i] == 'IND':
                        true_positive_ind += 1
                    else:
                        false_positive_ind += 1

                elif prediction == 'group':
                    if test_answers[i] == 'GRP':
                        true_positive_grp += 1
                    else:
                        false_positive_grp += 1

                elif prediction == 'other':
                    if test_answers[i] == 'OTH':
                        true_positive_oth += 1
                    else:
                        false_positive_oth += 1

        metrics = [(true_positive_not, false_positive_not, 'NOT'),
                   (true_positive_off, false_positive_off, 'OFF'),
                   (true_positive_unt, false_positive_unt, 'UNT'),
                   (true_positive_tin, false_positive_tin, 'TIN'),
                   (true_positive_ind, false_positive_ind, 'IND'),
                   (true_positive_grp, false_positive_grp, 'GRP'),
                   (true_positive_oth, false_positive_oth, 'OTH')]

        for metric in metrics:
            pp = metric[0] + metric[1]
            precision = metric[0] / pp if pp > 0 else 0
            recall = metric[0] / category_frequency[metric[2]]
            f1 = 2 * precision * recall / \
                (precision + recall) if precision + recall > 0 else 0

            print(f'{metric[2]}, {precision}, {recall}, {f1}')

        end = time.time()
        t = round(end - start, 2)

        if t > 60:
            duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
        else:
            duration = str(t) + 's'

        print(f'Time taken: {duration}')
        print()


if __name__ == '__main__':
    custom_models = [f for f in listdir(
        main_config.model_directory) if 'olid' in f or 'weak' in f]

    # Import unique filtered comments for testing
    filtered_comments = comment_filter.c_filter(
        input_list=main_config.test_tweets_getter(),
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False)

    evaluate_model(
        models=custom_models,
        test_comments=filtered_comments[:],
        test_answers=main_config.answers_getter(),
        offensive_threshold=0.3,
        targeted_threshold=0.05)
