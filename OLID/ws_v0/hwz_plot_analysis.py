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


def f1_score(p, r):
    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def evaluate(test_comments: list, undersampled: bool, models: list):

    figure, axis = plt.subplots(2, 2)

    unt_count = 117

    if undersampled:
        off_count = 502
        not_count = 289
        tin_count = 385
        ind_count = 177
        grp_count = 153
        oth_count = 55
        total_count = 791

    else:
        off_count = 234
        not_count = 234
        tin_count = 117
        ind_count = 39
        grp_count = 39
        oth_count = 39
        total_count = 468

    task_a_answers_array = np.concatenate([np.zeros(not_count), np.ones(off_count)])
    task_b_answers_array = np.concatenate([np.zeros(unt_count), np.ones(tin_count)])
    task_c_answers = ['IND'] * ind_count + ['GRP'] * grp_count + ['OTH'] * oth_count

    for index, model in enumerate(models):

        print(model)

        nlp = spacy.load(main_config.model_directory + model + '/model-best')
        docs = list(nlp.pipe(test_comments))

        task_a_predictions_array = np.array([doc.cats['offensive'] for doc in docs])
        task_b_predictions_array = np.array([docs[i].cats['targeted'] for i in range(not_count, total_count)])

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

            for p, r in zip(precision, recall):
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
                if macro_f1 > final_f1[-1]:
                    final_f1 = [f1, neg_f1, macro_f1]

            print(task)
            print(final_f1)

        tp_ind = 0
        fp_ind = 0

        tp_grp = 0
        fp_grp = 0

        tp_oth = 0
        fp_oth = 0

        for i in range(not_count + unt_count, total_count):
            result = docs[i].cats

            result.pop('offensive')
            result.pop('targeted')

            prediction = max(result, key=result.get)

            if prediction == 'individual':
                if task_c_answers[i - total_count] == 'IND':
                    tp_ind += 1
                else:
                    fp_ind += 1

            elif prediction == 'group':
                if task_c_answers[i - total_count] == 'GRP':
                    tp_grp += 1
                else:
                    fp_grp += 1

            else:
                if task_c_answers[i - total_count] == 'OTH':
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

        print('C')
        print([f1_ind, f1_grp, f1_oth, macro_f1])
        print()


if __name__ == '__main__':

    is_undersampled = True

    # HWZ evaluation
    filtered_hwz_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=999,
        uncased=False,
        unique=False,
        edmw=True,
        input_list=main_config.balanced_hwz_getter(is_undersampled))

    models = [f for f in listdir(main_config.model_directory) if 'olid' in f and 'uncased' not in f]

    evaluate(
        test_comments=filtered_hwz_comments,
        undersampled=is_undersampled,
        models=models)