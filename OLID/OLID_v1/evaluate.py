from collections import Counter
from os import listdir
import main_config
import spacy
import time

import numpy as np
from sklearn.metrics import precision_recall_curve

spacy.require_gpu()


def evaluate_model(
        models: list,
        cased_test_tweets: list,
        test_answers: list):

    category_frequency = Counter(test_answers)

    for model in models:

        start = time.time()

        print(model)

        if 'uncased' in model:
            all_tweets = [tweet.lower() for tweet in cased_test_tweets]
        else:
            all_tweets = cased_test_tweets

        nlp = spacy.load(main_config.model_directory + model + '/model-best')
        docs = list(nlp.pipe(all_tweets))

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

        for i in range(len(all_tweets)):
            result = docs[i].cats
            if i < 860:
                if result["offensive"] > 0.5:
                    if test_answers[i] == 'OFF':
                        true_positive_off += 1

                    else:
                        false_positive_off += 1

                elif test_answers[i] == 'NOT':
                    true_positive_not += 1

                else:
                    false_positive_not += 1

            elif i < 1100:
                if result["targeted"] > 0.5:
                    if test_answers[i] == 'TIN':
                        true_positive_tin += 1

                    else:
                        false_positive_tin += 1

                elif test_answers[i] == 'UNT':
                    true_positive_unt += 1

                else:
                    false_positive_unt += 1

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

                elif test_answers[i] == 'OTH':
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
        main_config.model_directory)]

    evaluate_model(
        models=custom_models,
        cased_test_tweets=main_config.preprocess(
            tweets=main_config.test_tweets_getter(), uncased=False),
        test_answers=main_config.answers_getter())
