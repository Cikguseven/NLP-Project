from bad_words import offensive_lexicon
from detoxify import Detoxify
from flair.data import Sentence
from flair.models import TextClassifier
from hatesonar import Sonar
from math import exp
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np
import target_classifier


def custom_sigmoid(x):
    return 1 / (1 + exp(2.5 * x + 0.125))


def sentiment_vader(sentence):
    return SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']


def sentiment_textblob(sentence):
    return TextBlob(sentence).sentiment.polarity


task_a_answers_array = np.concatenate([np.zeros(1102), np.ones(2392)])

task_b_answers_array = np.concatenate([np.zeros(551), np.ones(1841)])

task_c_answers = ['OTH'] * 430 + ['IND'] * 430 + ['GRP'] * 430


def evaluate(test_tweets: list, uncased_test_tweets: list):

    # Hate speech text classifier models from Hugging Face
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

    tc_8 = pipeline(task='text-classification', model="./pipelines/tc_8/",
                    tokenizer="./pipelines/tc_8/", device=0)

    sonar = Sonar()

    detoxify = Detoxify('unbiased', device='cuda')

    flair = TextClassifier.load('sentiment')

    models = [(tc_1, 'tc_1', 'POSITIVE', 'uncased', 1),
              (tc_2, 'tc_2', 'LABEL_0', 'cased', 2),
              (tc_3, 'tc_3', 'LABEL_0', 'cased', 3),
              (tc_4, 'tc_4', 'Non-Offensive', 'cased', 4),
              (tc_5, 'tc_5', 'LABEL_0', 'cased', 5),
              (tc_5, 'tc_5_inverse1', 'LABEL_0', 'cased', 6),
              (tc_5, 'tc_5_inverse12', 'LABEL_0', 'cased', 7),
              (tc_6, 'tc_6', 'NO_HATE', 'cased', 8),
              (tc_7, 'tc_7', 'NON_HATE', 'cased', 9),
              (tc_8, 'tc_8', 'POSITIVE', 'cased', 10),
              (vader, 'sentiment_vader', 'Non-Offensive', 'cased', 11),
              (textblob, 'sentiment_textblob', 'LABEL_0', 'cased', 12),
              (sonar, 'sonar_hate', 'NO_HATE', 'cased', 13),
              (sonar, 'sonar_hatf', 'NO_HATE', 'cased', 14),
              (sonar, 'sonar_ol', 'NO_HATE', 'cased', 15),
              (sonar, 'sonar_olf', 'NO_HATE', 'cased', 16),
              (detoxify, 'detoxify_toxicity', 'NON_HATE', 'cased', 17),
              (detoxify, 'detoxify_severe_toxicity', 'NON_HATE', 'cased', 18),
              (detoxify, 'detoxify_obscene', 'NON_HATE', 'cased', 19),
              (detoxify, 'detoxify_identity_attack', 'NON_HATE', 'cased', 20),
              (detoxify, 'detoxify_insult', 'NON_HATE', 'cased', 21),
              (detoxify, 'detoxify_threat', 'NON_HATE', 'cased', 22),
              (detoxify, 'detoxify_sexual_explicit', 'NON_HATE', 'cased', 23),
              (flair, 'flair', None, 'cased', 24)]

    figure, axis = plt.subplots(2, 2)

    for classifier, name, wrong_label, case, index in models:

        print(name)

        classifier_score = []

        if 'tc_' in name:
            if case == 'uncased':
                results = classifier(uncased_test_tweets)
            else:
                results = classifier(test_tweets)

            if 'inverse12' in name:
                classifier_score = [1 - result['score'] if result['label'] in ('LABEL_0', 'LABEL_1', 'LABEL_2') else result['score'] for result in results]
            elif 'inverse1' in name:
                classifier_score = [1 - result['score'] if result['label'] in ('LABEL_0', 'LABEL_1') else result['score'] for result in results]
            else:
                classifier_score = [1 - result['score'] if result['label'] == wrong_label else result['score'] for result in results]
                
        elif 'vader' in name:
            classifier_score = [custom_sigmoid(vader(tweet)) for tweet in test_tweets]

        elif 'textblob' in name:
                classifier_score = [custom_sigmoid(textblob(tweet)) for tweet in test_tweets]

        elif 'sonar' in name:
            sonar_inverse_neither_score = np.array([1 - \
                classifier.ping(text=tweet)['classes'][2]['confidence'] for tweet in test_tweets])
            sonar_hate_score = np.array([classifier.ping(text=tweet)['classes'][0]['confidence'] for tweet in test_tweets])
            sonar_offensive_score = np.array([classifier.ping(text=tweet)['classes'][1]['confidence'] for tweet in test_tweets])

            if 'hatf' in name:
                classifier_score = sonar_hate_score / sonar_inverse_neither_score
            elif 'olf' in name:
                classifier_score = sonar_offensive_score / sonar_inverse_neither_score
            elif 'hate' in name:
                classifier_score = sonar_hate_score
            elif 'ol' in name:
                classifier_score = sonar_offensive_score
            
        elif 'detoxify' in name:
            classifier_score = [classifier.predict(tweet)[name.replace('detoxify_', '')] for tweet in test_tweets]

        elif 'flair' in name:
            for tweet in test_tweets:
                sentence = Sentence(tweet)
                classifier.predict(sentence)
                flair_result = sentence.labels[0].to_dict()
                flair_score = 1 - flair_result['confidence'] if flair_result['value'] == 'POSITIVE' else flair_result['confidence']
                classifier_score.append(flair_score)

        task_a_predictions_array = np.array(classifier_score)
        task_b_predictions_array = task_a_predictions_array[1102:]

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
                axis[1, x].set_title("Task A ROC")
                axis[1, x].set_xlabel("False Positive Rate")
                axis[1, x].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_binary(test_tweets: list, uncased_test_tweets: list):

    tc_9 = pipeline(task='text2text-generation', model="./pipelines/tc_9/",
                    tokenizer="./pipelines/tc_9/", device=0)

    # binary_models = [(tc_9, 'tc_9', 'no-hate-speech'),
    #                  (offensive_lexicon, 'offensive_lexicon', 'LABEL_0'),
    #                  (target_classifier.weak_classifier, 'spacy', None)]

    binary_models = [(target_classifier.weak_classifier, 'spacy', None)]

    for classifier, name, wrong_label in binary_models:
        if 'spacy' in name:
            results = classifier(test_tweets[2204:])

            cm = confusion_matrix(task_c_answers, results, labels=["OTH", "IND", "GRP"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["OTH", "IND", "GRP"])

            disp.plot(values_format='')
            plt.show()
            
        else:
            if '9' in name:
                results = classifier(test_tweets)
                classifier_score = [0 if result['generated_text'] == wrong_label else 1 for result in results]
                predictions_array = np.array(classifier_score)

            elif 'lexicon' in name:
                classifier_score = [1 if any(offensive_word in uncased_tweet for offensive_word in offensive_lexicon) else 0 for uncased_tweet in uncased_test_tweets]
                predictions_array = np.array(classifier_score)

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
    olid_balanced_tweets = main_config.balanced_tweets_getter()

    # Import unique filtered comments for testing
    filtered_tweets = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=olid_balanced_tweets)

    uncased_tweets = [tweet.lower() for tweet in filtered_tweets]

    # evaluate(
    #     test_tweets=filtered_tweets[:2204],
    #     uncased_test_tweets=uncased_tweets[:2204])

    evaluate_binary(
        test_tweets=filtered_tweets,
        uncased_test_tweets=uncased_tweets)
