from transformers import pipeline
from sklearn.metrics import precision_recall_curve, roc_curve
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np

tc_5 = pipeline(task='text-classification', model="./pipelines/tc_5/",
                tokenizer="./pipelines/tc_5/", device=0)

def evaluate(
        test_tweets: list):

    task_a_answers_array = np.array(
        [0 if x < 1048 else 1 for x in range(2096)])
    task_b_answers_array = np.array(
        [0 if x < 524 else 1 for x in range(1048)])
    oth_answers_array = np.array(
        [1 if x < 430 else 0 for x in range(1290)])
    ind_answers_array = np.array(
        [1 if 429 < x < 860 else 0 for x in range(1290)])
    grp_answers_array = np.array(
        [1 if x > 859 else 0 for x in range(1290)])

    figure, axis = plt.subplots(2, 2)

    LABEL_0 = np.zeros(2096)
    INVERSE_0 = np.zeros(2096)
    INVERSE_0_1 = np.zeros(2096)
    INVERSE_0_1_2 = np.zeros(2096)
    LABEL_1 = np.zeros(2096)
    LABEL_2 = np.zeros(2096)
    LABEL_3 = np.zeros(2096)
    
    results = tc_5(test_tweets)

    for i in range(len(results)):
        if results[i]['label'] == 'LABEL_0':
            LABEL_0[i] = results[i]['score']
            INVERSE_0[i] = 1 - results[i]['score']
            INVERSE_0_1[i] = 1 - results[i]['score']
            INVERSE_0_1_2[i] = 1 - results[i]['score']
        elif results[i]['label'] == 'LABEL_1':
            LABEL_1[i] = results[i]['score']
            INVERSE_0_1[i] = 1 - results[i]['score']
            INVERSE_0_1_2[i] = 1 - results[i]['score']
        elif results[i]['label'] == 'LABEL_2':
            LABEL_2[i] = results[i]['score']
            INVERSE_0_1_2[i] = 1 - results[i]['score']
        else:
            LABEL_3[i] = results[i]['score']


    print(LABEL_0[:10])
    print(INVERSE_0[:10])
    print(INVERSE_0_1[:10])
    print(INVERSE_0_1_2[:10])
    print(LABEL_1[:10])
    print(LABEL_2[:10])
    print(LABEL_3[:10])

    TASK_A = INVERSE_0 + LABEL_1 + LABEL_2 + LABEL_3
    TASK_B = INVERSE_0_1 + LABEL_2 + LABEL_3
    NEW_TASK_B = INVERSE_0_1_2 + LABEL_3

    keys = [('TASK_A', TASK_A), ('TASK_B', TASK_B), ('NEW_TASK_B', NEW_TASK_B)]

    for key, score in keys:

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
