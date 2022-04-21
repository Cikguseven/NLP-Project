# Do analysis for pipe 5

from hatesonar import Sonar
from sklearn.metrics import precision_recall_curve, roc_curve
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np

sonar_model = Sonar()


def evaluate(
        test_tweets: list):

    task_a_answers_array = np.array(
        [0 if x < 1048 else 1 for x in range(2096)])
    task_b_answers_array = np.array(
        [0 if x < 524 else 1 for x in range(1048)])

    figure, axis = plt.subplots(2, 2)

    sonar_hate = []
    sonar_offensive = []
    sonar_either = []
    sonar_hateovereither = []

    keys = [(sonar_hate, 'sonar_hate'), (sonar_offensive, 'sonar_offensive'), (sonar_either,'sonar_either'), (sonar_hateovereither, 'sonar_hateovereither')]

    for tweet in test_tweets:
        sonar_score = sonar_model.ping(text=tweet)['classes']

        hate = sonar_score[0]['confidence']
        either = 1 - sonar_score[2]['confidence']

        sonar_hate.append(hate)
        sonar_offensive.append(sonar_score[1]['confidence'])
        sonar_either.append(either)
        sonar_hateovereither.append(hate / either)

    for score, key in keys:

        task_a_predictions_array = np.array(score)
        task_b_predictions_array = task_a_predictions_array[1048:]

        precision, recall, thresholds = precision_recall_curve(
            task_a_answers_array, task_a_predictions_array)
        axis[0, 0].plot(recall, precision, label=key)
        axis[0, 0].set_title("Task A PRC")
        axis[0, 0].set_xlabel("Recall")
        axis[0, 0].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_a_answers_array, task_a_predictions_array)
        axis[1, 0].plot(fpr, recall, label=key)
        axis[1, 0].set_title("Task A ROC")
        axis[1, 0].set_xlabel("False Positive Rate")
        axis[1, 0].set_ylabel("True Positive Rate")

        precision, recall, thresholds = precision_recall_curve(
            task_b_answers_array, task_b_predictions_array)
        axis[0, 1].plot(recall, precision, label=key)
        axis[0, 1].set_title("Task B PRC")
        axis[0, 1].set_xlabel("Recall")
        axis[0, 1].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_b_answers_array, task_b_predictions_array)
        axis[1, 1].plot(fpr, recall, label=key)
        axis[1, 1].set_title("Task B ROC")
        axis[1, 1].set_xlabel("False Positive Rate")
        axis[1, 1].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    olid_training_data = main_config.training_tweets_getter()

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

    evaluate(
        test_tweets=filtered_tweets)
