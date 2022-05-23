from os import listdir
from sklearn.metrics import precision_recall_curve
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np
import spacy
import sys

spacy.require_gpu()
np.set_printoptions(threshold=sys.maxsize)


def f1_score(p, r):
    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def evaluate(test_tweets: list, test_answers: list, balanced: bool, models: list):

    if balanced:
        task_a_answers_array = np.concatenate([np.zeros(1102), np.ones(1102)])
        task_b_answers_array = np.concatenate([np.zeros(551), np.ones(551)])
        task_c_answers = ['OTH'] * 430 + ['IND'] * 430 + ['GRP'] * 430

        a_limit = 2204
        b_lower = 1102
        b_limit = 2204
        c_limit = 3494

        off_count = 1102
        not_count = 1102
        tin_count = 551
        unt_count = 551
        ind_count = 430
        grp_count = 430
        oth_count = 430

    else:
        task_a_answers_array = np.array([1 if x == 'OFF' else 0 for x in test_answers[:860]])
        task_b_answers_array = np.array([1 if x == 'TIN' else 0 for x in test_answers[860:1100]])
        task_c_answers = test_answers[1100:]

        a_limit = 860
        b_lower = 860
        b_limit = 1100
        c_limit = 1313

        off_count = 240
        not_count = 620
        tin_count = 213
        unt_count = 27
        ind_count = 100
        grp_count = 78
        oth_count = 35

    for index, model in enumerate(models):

        print(model)

        nlp = spacy.load(main_config.model_directory + model + '/model-best')
        docs = list(nlp.pipe(test_tweets))

        task_a_predictions_array = np.array([docs[i].cats['offensive'] for i in range(a_limit)])
        task_b_predictions_array = np.array([docs[i].cats['targeted'] for i in range(b_lower, b_limit)])

        tasks = ['A', 'B']

        for task in tasks:
            if task == 'A':
                precision, recall, thresholds = precision_recall_curve(task_a_answers_array, task_a_predictions_array)
                total_pos = off_count
                total_neg = not_count
            else:
                precision, recall, thresholds = precision_recall_curve(
                    task_b_answers_array, task_b_predictions_array)
                total_pos = tin_count
                total_neg = unt_count

            precision = precision[:-1]
            recall = recall[:-1]

            final_f1 = [0, 0, 0]
            final_t = 0

            macro_f1_array = []
            threshold_array = []

            for p, r, t in zip(precision, recall, thresholds):
                f1 = f1_score(p, r)
                tp_count = int(r * total_pos)
                fp_count = tp_count // p - tp_count
                tn_count = total_neg - fp_count
                fn_count = int((1 - r) * total_pos)
                if tn_count > 0:
                    neg_p = tn_count / (tn_count + fn_count)
                    neg_r = tn_count / total_neg
                    neg_f1 = f1_score(neg_p, neg_r)
                else:
                    neg_f1 = 0
                macro_f1 = (f1  + neg_f1) / 2

                macro_f1_array.append(macro_f1)
                threshold_array.append(t)

                if macro_f1 > final_f1[-1]:
                    final_f1 = [f1, neg_f1, macro_f1]
                    final_t = t

            print(final_f1)
            print(final_t)

            plt.figure()
            x_threshold = np.array(threshold_array)
            y_f1 = np.array(macro_f1_array)

            plt.plot(x_threshold, y_f1)
            plt.show()

        tp_ind = 0
        fp_ind = 0

        tp_grp = 0
        fp_grp = 0

        tp_oth = 0
        fp_oth = 0

        for i in range(b_limit, c_limit):
            result = docs[i].cats

            result.pop('offensive')
            result.pop('targeted')

            prediction = max(result, key=result.get)

            if prediction == 'individual':
                if task_c_answers[i - c_limit] == 'IND':
                    tp_ind += 1
                else:
                    fp_ind += 1

            elif prediction == 'group':
                if task_c_answers[i - c_limit] == 'GRP':
                    tp_grp += 1
                else:
                    fp_grp += 1

            else:
                if task_c_answers[i - c_limit] == 'OTH':
                    tp_oth += 1
                else:
                    fp_oth += 1

        precision_ind = tp_ind / (tp_ind + fp_ind)
        recall_ind = tp_ind / ind_count
        f1_ind = f1_score(precision_ind, recall_ind) 

        precision_grp = tp_grp / (tp_grp + fp_grp)
        recall_grp = tp_grp / grp_count
        f1_grp = f1_score(precision_grp, recall_grp) 


        if tp_oth + fp_oth > 0:
            precision_oth = tp_oth / (tp_oth + fp_oth)
            recall_oth = tp_oth / oth_count
            f1_oth = f1_score(precision_oth, recall_oth)
        else:
            f1_oth = 0

        macro_f1 = (f1_ind + f1_grp + f1_oth) / 3

        print([f1_ind, f1_grp, f1_oth, macro_f1])
        print()


if __name__ == '__main__':

    # OLID evaluation
    
    use_balanced_olid = False

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

    models = [f for f in listdir(main_config.model_directory) if 'wk14_ws_v1_60a_10b_lexicon1_tc9removed_13240' in f]

    evaluate(
        test_tweets=filtered_tweets[:],
        test_answers=main_config.answers_getter(),
        balanced=use_balanced_olid,
        models=models)
