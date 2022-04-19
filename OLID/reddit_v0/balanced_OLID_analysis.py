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

spacy.require_gpu()


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


def random_seed():
    return 0.1346365467


if __name__ == '__main__':
    olid_training_data = main_config.training_tweets_getter()

    sample_tweets = []

    nn_tweets = []
    nn_counter = 0

    ou_tweets = []
    ou_counter = 0

    ot_tweets = []
    ot_counter = 0

    for tweet in olid_training_data:
        if tweet[2] == 'NOT' and nn_counter < 1048:
            nn_tweets.append(tweet[1])
            nn_counter += 1
        elif tweet[3] == 'UNT':
            ou_tweets.append(tweet[1])
            ou_counter += 1
        elif ot_counter < 524:
            ot_tweets.append(tweet[1])
            ot_counter += 1

    sample_tweets = nn_tweets + ou_tweets + ot_tweets

    # Import unique filtered comments for testing
    filtered_tweets = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=sample_tweets)

    evaluate_model(
        test_tweets=filtered_tweets)
