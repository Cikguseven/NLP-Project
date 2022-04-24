from os import listdir
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np
import random
import spacy
import time
import weak_signals

def random_seed():
    return 0.1346365467


def evaluate_model(
        test_tweets: list):

    task_a_answers_array = np.array(
        [0 if x < 1048 else 1 for x in range(2096)])
    task_b_answers_array = np.array(
        [0 if x < 524 else 1 for x in range(1048)])

    uncased_test_tweets = [tweet.lower() for tweet in test_tweets]

    figure, axis = plt.subplots(2, 2)

    task_a_predictions_array = weak_signals.model_aggregator(test_tweets, uncased_test_tweets, 'a')
    task_b_predictions_array = weak_signals.model_aggregator(test_tweets[1048:], uncased_test_tweets[1048:], 'b')

    precision, recall, thresholds = precision_recall_curve(
        task_a_answers_array, task_a_predictions_array)
    axis[0, 0].plot(recall, precision)
    axis[0, 0].set_title("Task A PRC")
    axis[0, 0].set_xlabel("Recall")
    axis[0, 0].set_ylabel("Precision")

    fpr, recall, thresholds = roc_curve(
        task_a_answers_array, task_a_predictions_array)
    axis[1, 0].plot(fpr, recall)
    axis[1, 0].set_title("Task A ROC")
    axis[1, 0].set_xlabel("False Positive Rate")
    axis[1, 0].set_ylabel("True Positive Rate")

    precision, recall, thresholds = precision_recall_curve(
        task_b_answers_array, task_b_predictions_array)
    axis[0, 1].plot(recall, precision)
    axis[0, 1].set_title("Task B PRC")
    axis[0, 1].set_xlabel("Recall")
    axis[0, 1].set_ylabel("Precision")

    fpr, recall, thresholds = roc_curve(
        task_b_answers_array, task_b_predictions_array)
    axis[1, 1].plot(fpr, recall)
    axis[1, 1].set_title("Task B ROC")
    axis[1, 1].set_xlabel("False Positive Rate")
    axis[1, 1].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    spacy.require_gpu()

    olid_training_data = main_config.balanced_tweets_getter(analysis_set=True)

    # Import unique filtered comments for testing
    filtered_tweets = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=olid_training_data)

    evaluate_model(
        test_tweets=filtered_tweets)
