from bad_words import offensive_lexicon
from os import listdir
from sklearn.metrics import precision_recall_curve, roc_curve, ConfusionMatrixDisplay
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np
import spacy
import sys

spacy.require_gpu()
np.set_printoptions(threshold=sys.maxsize)

def evaluate(test_tweets: list, test_answers: list, balanced: bool, models: list):

    figure, axis = plt.subplots(2, 2)

    if balanced:
        task_a_answers_array = np.concatenate([np.zeros(1102), np.ones(2392)])
        task_b_answers_array = np.concatenate([np.zeros(551), np.ones(1841)])
        task_c_answers = ['OTH'] * 430 + ['IND'] * 430 + ['GRP'] * 430

        a_limit = 3494
        b_lower = 1102
        b_limit = a_limit
        c_lower = 2204
        c_limit = a_limit
        off_count = 2392
        not_count = 1102
        tin_count = 1841

    else:
        task_a_answers_array = np.array([1 if x == 'OFF' else 0 for x in test_answers[:860]])
        task_b_answers_array = np.array([1 if x == 'TIN' else 0 for x in test_answers[860:1100]])
        task_c_answers = test_answers[1100:]

        a_limit = 860
        b_lower = a_limit
        b_limit = 1100
        c_lower = b_limit
        c_limit = 1313

    for index, model in enumerate(models):

        print(model)

        nlp = spacy.load(main_config.model_directory + model + '/model-best')
        docs = list(nlp.pipe(test_tweets))

        task_a_predictions_array = np.array([docs[i].cats['offensive'] for i in range(a_limit)])
        task_b_predictions_array = np.array([docs[i].cats['targeted'] for i in range(b_lower, b_limit)])

        for i in range(c_lower, c_limit):
            result = docs[i].cats

            result.pop('offensive')
            result.pop('targeted')

            prediction = max(result, key=result.get)

            if prediction == 'individual':
                results.append('IND')
                if task_c_answers[i - c_limit] == 'IND':
                    true_positive_ind += 1
                else:
                    false_positive_ind += 1

            elif prediction == 'group':
                results.append('GRP')
                if task_c_answers[i - c_limit] == 'GRP':
                    true_positive_grp += 1
                else:
                    false_positive_grp += 1

            else:
                results.append('OTH')
                if task_c_answers[i - c_limit] == 'OTH':
                    true_positive_oth += 1
                else:
                    false_positive_oth += 1

        tasks = ['A', 'B']

        for task in tasks:
            if task == 'A':
                precision, recall, thresholds = precision_recall_curve(task_a_answers_array, task_a_predictions_array)
                pos_count = off_count
            else:
                precision, recall, thresholds = precision_recall_curve(
                    task_b_answers_array, task_b_predictions_array)

            del precision[-1]
            del recall[-1]

            macro_f1 = (0, 0, 0, 0)

            for p, r in zip(precision, recall):
                normal_f1 = 2 * p * r / (p + r)
                tp_count = r * total_pos




   

            true_positive_not = 0
            false_positive_not = 0

            true_positive_off = 0
            false_positive_off = 0

            true_positive_unt = 0
            false_positive_unt = 0

            true_positive_tin = 0
            false_positive_tin = 0

            for i in range(3494):
                if i < 1102:
                    if predictions_array[i] == 0:
                        true_positive_not += 1
                    else:
                        false_positive_off += 1

                else:
                    if predictions_array[i]:
                        true_positive_off += 1
                    else:
                        false_positive_not += 1

                    if i < 1653:
                        if predictions_array[i] == 0:
                            true_positive_unt += 1
                        else:
                            false_positive_tin += 1

                    else:
                        if predictions_array[i]:
                            true_positive_tin += 1
                        else:
                            false_positive_unt += 1

            metrics = [(true_positive_not, false_positive_not, 'NOT', 1102),
                       (true_positive_off, false_positive_off, 'OFF', 2392),
                       (true_positive_unt, false_positive_unt, 'UNT', 551),
                       (true_positive_tin, false_positive_tin, 'TIN', 1841)]

            for metric in metrics:
                pp = metric[0] + metric[1]
                precision = metric[0] / pp if pp > 0 else 0
                recall = metric[0] / metric[3]
                f1 = 2 * precision * recall / \
                    (precision + recall) if precision + recall > 0 else 0

                print(f'{metric[2]}, {precision}, {recall}, {f1}')


if __name__ == '__main__':

    use_balanced_olid = True

    if use_balanced_olid:
        get_tweets = main_config.balanced_tweets_getter(analysis_set=True)
    else:
        get_tweets = main_config.test_tweets_getter()

    # Import unique filtered comments for testing
    filtered_tweets = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=get_tweets)

    models = [f for f in listdir(main_config.model_directory) if 'b_' in f and 'uncased' not in f]

    evaluate(
        test_tweets=filtered_tweets[:],
        test_answers=main_config.answers_getter(),
        balanced=use_balanced_olid,
        models=models)

    evaluate_c(
        test_tweets=filtered_tweets[:],
        test_answers=main_config.answers_getter(),
        balanced=use_balanced_olid,
        models=models)
