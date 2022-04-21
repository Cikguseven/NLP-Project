from bad_words import offensive_lexicon
from detoxify import Detoxify
from flair.data import Sentence
from flair.models import TextClassifier
from hatesonar import Sonar
from math import exp
from sklearn.metrics import precision_recall_curve, roc_curve
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np

# Hate speech text classifier models from Hugging Face
tc_0 = pipeline(task='text-classification', model="./pipelines/tc_0/",
                tokenizer="./pipelines/tc_0/", device=0)

tc_1 = pipeline(task='text-classification', model="./pipelines/tc_1/",
                tokenizer="./pipelines/tc_1/", device=0)

tc_2 = pipeline(task='text-classification', model="./pipelines/tc_2/",
                tokenizer="./pipelines/tc_2/", device=0)

tc_3 = pipeline(task='text-classification', model="./pipelines/tc_3/",
                tokenizer="./pipelines/tc_3/", device=0)

tc_4 = pipeline(task='text-classification', model="./pipelines/tc_4/",
                tokenizer="./pipelines/tc_4/", device=0)

tc_5 = pipeline(task='text-classification', model="./pipelines/tc_5/",
                tokenizer="./pipelines/tc_5/", device=0)

tc_6 = pipeline(task='text-classification', model="./pipelines/tc_6/",
                tokenizer="./pipelines/tc_6/", device=0)

tc_7 = pipeline(task='text-classification', model="./pipelines/tc_7/",
                tokenizer="./pipelines/tc_7/", device=0)

tc_8 = pipeline(task='text2text-generation', model="./pipelines/tc_8/",
                tokenizer="./pipelines/tc_8/", device=0)


def custom_sigmoid(x):
    return 1 / (1 + exp(2.5 * x + 0.125))


def sentiment_vader(sentence):
    return SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']


def sentiment_textblob(sentence):
    return TextBlob(sentence).sentiment.polarity


sonar = Sonar()

detoxify = Detoxify('unbiased', device='cuda')

flair = TextClassifier.load('sentiment')

models = [(tc_0, 'tc_0', 'POSITIVE', 'cased', 1),
          (tc_1, 'tc_1', 'POSITIVE', 'uncased', 2),
          (tc_2, 'tc_2', 'LABEL_0', 'cased', 3),
          (tc_3, 'tc_3', 'LABEL_0', 'cased', 4),
          (tc_4, 'tc_4', 'Non-Offensive', 'cased', 5),
          (tc_5, 'tc_5', 'LABEL_0', 'cased', 6),
          (tc_6, 'tc_6', 'NO_HATE', 'cased', 7),
          (tc_7, 'tc_7', 'NON_HATE', 'cased', 8),
          (tc_8, 'tc_8', 'no-hate-speech', 'cased', 9),
          (offensive_lexicon, 'offensive_lexicon', 'LABEL_0', 'cased', 10),
          (sentiment_vader, 'sentiment_vader', 'Non-Offensive', 'cased', 11),
          (sentiment_textblob, 'sentiment_textblob', 'LABEL_0', 'cased', 12),
          (sonar, 'sonar', 'NO_HATE', 'cased', 13),
          (detoxify, 'detoxify', 'NON_HATE', 'cased', 14),
          (flair, 'flair', 'no-hate-speech', 'cased', 15)]


def evaluate(
        test_tweets: list,
        test_tweets_c: list):

    task_a_answers_array = np.array(
        [0 if x < 1102 else 1 for x in range(2204)])
    task_b_answers_array = np.array(
        [0 if x < 551 else 1 for x in range(1102)])
    oth_answers_array = np.array(
        [1 if x < 430 else 0 for x in range(1290)])
    ind_answers_array = np.array(
        [1 if 429 < x < 860 else 0 for x in range(1290)])
    grp_answers_array = np.array(
        [1 if x > 859 else 0 for x in range(1290)])

    uncased_test_tweets = [tweet.lower() for tweet in test_tweets]
    uncased_test_tweets_c = [tweet.lower() for tweet in test_tweets_c]

    figure, axis = plt.subplots(2, 5)

    for classifier, name, wrong_label, case, index in models:

        classifier_score_b = None

        if 'tc_' in name:
            if case == 'uncased':
                results = classifier(uncased_test_tweets)
            else:
                results = classifier(test_tweets)

            if wrong_label == 'no-hate-speech':
                classifier_score_a = [0 if result['generated_text'] == wrong_label else 1 for result in results]
            else:
                classifier_score_a = [1 - result['score'] if result['label'] == wrong_label else result['score'] for result in results]

            if name == 'tc_5':
                classifier_score_b = np.array([1 - result['score'] if result['label'] in ('LABEL_0', 'LABEL_1') else result['score'] for result in results])

        elif 'offensive' in name:
            classifier_score_a = [1 if any(offensive_word in uncased_tweet for offensive_word in offensive_lexicon) else 0 for uncased_tweet in uncased_test_tweets]

        elif 'sentiment' in name:
            if 'vader' in name:
                classifier_score_a = [custom_sigmoid(sentiment_vader(tweet)) for tweet in test_tweets]
            else:
                classifier_score_a = [custom_sigmoid(sentiment_textblob(tweet)) for tweet in test_tweets]

        elif 'sonar' in name:
            classifier_score_a = [1 - \
                classifier.ping(text=tweet)['classes'][2]['confidence'] for tweet in test_tweets]
            classifier_score_b = np.array([classifier.ping(text=tweet)['classes'][0]['confidence'] for tweet in test_tweets]) / np.array(classifier_score_a)

        elif 'detoxify' in name:
            classifier_score_a = [classifier.predict(tweet)['toxicity'] for tweet in test_tweets]
            classifier_score_b = np.array([classifier.predict(tweet)['insult'] for tweet in test_tweets])

        elif 'flair' in name:
            classifier_score_a = []
            for tweet in test_tweets:
                sentence = Sentence(tweet)
                classifier.predict(sentence)
                flair_result = sentence.labels[0].to_dict()
                flair_score = 1 - flair_result['confidence'] if flair_result['value'] == 'POSITIVE' else flair_result['confidence']
                classifier_score_a.append(flair_score)      

        task_a_predictions_array = np.array(classifier_score_a)

        if type(classifier_score_b) == list:
            task_b_predictions_array = classifier_score_b
        else:
            task_b_predictions_array = task_a_predictions_array[1048:]

        precision, recall, thresholds = precision_recall_curve(
            task_a_answers_array, task_a_predictions_array)
        if index > 9:
            axis[0, 0].plot(recall, precision, label=name, ls=':')
        else:
            axis[0, 0].plot(recall, precision, label=name)
        axis[0, 0].set_title("Task A PRC")
        axis[0, 0].set_xlabel("Recall")
        axis[0, 0].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_a_answers_array, task_a_predictions_array)
        if index > 9:
            axis[1, 0].plot(fpr, recall, label=name, ls=':')
        else:
            axis[1, 0].plot(fpr, recall, label=name)
        axis[1, 0].set_title("Task A ROC")
        axis[1, 0].set_xlabel("False Positive Rate")
        axis[1, 0].set_ylabel("True Positive Rate")

        precision, recall, thresholds = precision_recall_curve(
            task_b_answers_array, task_b_predictions_array)
        if index > 9:
            axis[0, 1].plot(recall, precision, label=name, ls=':')
        else:
            axis[0, 1].plot(recall, precision, label=name)
        axis[0, 1].set_title("Task B PRC")
        axis[0, 1].set_xlabel("Recall")
        axis[0, 1].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_b_answers_array, task_b_predictions_array)
        if index > 9:
            axis[1, 1].plot(fpr, recall, label=name, ls=':')
        else:
            axis[1, 1].plot(fpr, recall, label=name)
        axis[1, 1].set_title("Task B ROC")
        axis[1, 1].set_xlabel("False Positive Rate")
        axis[1, 1].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    task_a_b, task_c = main_config.balanced_tweets_getter()

    # Import unique filtered comments for testing
    filtered_a_b = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=task_a_b)

    filtered_c = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=task_c)

    evaluate(
        test_tweets=filtered_a_b,
        test_tweets_c=filtered_c)
