from os import listdir
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
import sys

spacy.require_gpu()
np.set_printoptions(threshold=sys.maxsize)


def evaluate(test_tweets: list, test_answers: list, balanced: bool, models: list):

    figure, axis = plt.subplots(2, 2)

    if balanced:
        task_a_answers_array = np.concatenate([np.zeros(1102), np.ones(1102)])
        task_b_answers_array = np.concatenate([np.zeros(551), np.ones(551)])
        task_c_answers = ['OTH'] * 430 + ['IND'] * 430 + ['GRP'] * 430

        a_limit = 2204
        b_lower = 1102
        b_limit = 2204

    else:
        task_a_answers_array = np.array([1 if x == 'OFF' else 0 for x in test_answers[:860]])
        task_b_answers_array = np.array([1 if x == 'TIN' else 0 for x in test_answers[860:1100]])

        a_limit = 860
        b_lower = 860
        b_limit = 1100

    for index, model in enumerate(models):
       
        print(model)

        name = model

        nlp = spacy.load(main_config.model_directory + model + '/model-best')
        docs = list(nlp.pipe(test_tweets))

        task_a_predictions_array = np.array([docs[i].cats['offensive'] for i in range(a_limit)])
        task_b_predictions_array = np.array([docs[i].cats['targeted'] for i in range(b_lower, b_limit)])

        curve_data = [('A', 'PRC'), ('A', 'ROC'), ('B', 'PRC'), ('B', 'ROC')]

        for task, curve in curve_data:
            if curve == 'PRC':
                if task == 'A':
                    precision, recall, thresholds = precision_recall_curve(task_a_answers_array, task_a_predictions_array)
                    x = 0
                else:
                    precision, recall, thresholds = precision_recall_curve(
                        task_b_answers_array, task_b_predictions_array)
                    x = 1

                if index > 19:
                    axis[0, x].plot(recall, precision, label=name, ls=':')
                elif index > 9:
                    axis[0, x].plot(recall, precision, label=name, ls='--')
                else:
                    axis[0, x].plot(recall, precision, label=name)
                axis[0, x].set_title("Task " + task + " PRC")
                axis[0, x].set_xlabel("Recall")
                axis[0, x].set_ylabel("Precision")

            else:
                if task == 'A':
                    fpr, recall, thresholds = roc_curve(task_a_answers_array, task_a_predictions_array)
                    x = 0
                else:
                    fpr, recall, thresholds = roc_curve(task_b_answers_array, task_b_predictions_array)
                    x = 1

                if index > 19:
                    axis[1, x].plot(fpr, recall, label=name, ls=':')
                elif index > 9:
                    axis[1, x].plot(fpr, recall, label=name, ls='--')
                else:
                    axis[1, x].plot(fpr, recall, label=name)
                axis[1, x].set_title("Task " + task + " ROC")
                axis[1, x].set_xlabel("False Positive Rate")
                axis[1, x].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_c(test_tweets: list, test_answers: list, balanced: bool, model: str):

    if balanced:
        task_c_answers = ['OTH'] * 430 + ['IND'] * 430 + ['GRP'] * 430
        test_tweets = test_tweets[-1290:]
    else:
        task_c_answers = test_answers[1100:]
        test_tweets = test_tweets[1100:]

    nlp = spacy.load(main_config.model_directory + model + '/model-best')
    docs = list(nlp.pipe(test_tweets))


    results = []

    for doc in docs:
        result = doc.cats
        result.pop('offensive')
        result.pop('targeted')

        prediction = max(result, key=result.get)

        if prediction == 'individual':
            results.append('IND')

        elif prediction == 'group':
            results.append('GRP')

        else:
            results.append('OTH')

    cm = confusion_matrix(task_c_answers, results, labels=["OTH", "IND", "GRP"], normalize='true')

    disp = ConfusionMatrixDisplay(cm, display_labels=["OTH", "IND", "GRP"])

    disp.plot()
    plt.show()


if __name__ == '__main__':

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

    spacy_models = [f for f in listdir(main_config.model_directory) if 'reddit' not in f]

    evaluate(
        test_tweets=filtered_tweets[:],
        test_answers=main_config.answers_getter(),
        balanced=use_balanced_olid,
        models=spacy_models)

    # evaluate_c(
    #     test_tweets=filtered_tweets[:],
    #     test_answers=main_config.answers_getter(),
    #     balanced=use_balanced_olid,
    #     model='ws_v1_50a_10b_lexicon10_tc9removed')