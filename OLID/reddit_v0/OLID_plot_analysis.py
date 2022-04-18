from os import listdir
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np
import spacy
import time
import weak_signals

spacy.require_gpu()


def evaluate_model(
        models: list,
        test_tweets: list,
        test_answers: list):

    task_a_answers_array = np.array(
        [1 if x == 'OFF' else 0 for x in test_answers[:860]])
    task_b_answers_array = np.array(
        [1 if x == 'TIN' else 0 for x in test_answers[860:1100]])

    uncased_test_tweets = [tweet.lower() for tweet in test_tweets]

    figure, axis = plt.subplots(2, 2)

    for model in models:

        print(model)

        if 'weak_signals_function' in model:
            task_a_predictions_array = weak_signals.model_aggregator(test_tweets[:860], uncased_test_tweets[:860], 'a')
            task_b_predictions_array = weak_signals.model_aggregator(test_tweets[860:1100], uncased_test_tweets[860:1100], 'b')

        else:
            nlp = spacy.load(main_config.model_directory +
                             model + '/model-best')

            if 'uncased' in model:
                docs = list(nlp.pipe(uncased_test_tweets))
            else:
                docs = list(nlp.pipe(test_tweets))

            task_a_predictions_array = np.array([docs[i].cats['offensive'] for i in range(860)])
            task_b_predictions_array = np.array([docs[i].cats['targeted'] for i in range(860, 1100)])

        precision, recall, thresholds = precision_recall_curve(
            task_a_answers_array, task_a_predictions_array)
        axis[0, 0].plot(recall, precision, label=model)
        axis[0, 0].set_title("Task A PRC")
        axis[0, 0].set_xlabel("Recall")
        axis[0, 0].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_a_answers_array, task_a_predictions_array)
        axis[1, 0].plot(fpr, recall, label=model)
        axis[1, 0].set_title("Task A ROC")
        axis[1, 0].set_xlabel("False Positive Rate")
        axis[1, 0].set_ylabel("True Positive Rate")

        precision, recall, thresholds = precision_recall_curve(
            task_b_answers_array, task_b_predictions_array)
        axis[0, 1].plot(recall, precision, label=model)
        axis[0, 1].set_title("Task B PRC")
        axis[0, 1].set_xlabel("Recall")
        axis[0, 1].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_b_answers_array, task_b_predictions_array)
        axis[1, 1].plot(fpr, recall, label=model)
        axis[1, 1].set_title("Task B ROC")
        axis[1, 1].set_xlabel("False Positive Rate")
        axis[1, 1].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    custom_models = [f for f in listdir(
        main_config.model_directory)]

    specific_model = ['weak_signals_function']

    all_models = custom_models + specific_model

    # Import unique filtered comments for testing
    filtered_tweets = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=main_config.test_tweets_getter())

    evaluate_model(
        models=all_models,
        test_tweets=filtered_tweets[:],
        test_answers=main_config.answers_getter())
