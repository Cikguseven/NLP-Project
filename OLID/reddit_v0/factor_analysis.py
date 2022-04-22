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
import test

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
          (tc_5, 'tc_5_inverse1', 'LABEL_0', 'cased', 7),
          (tc_5, 'tc_5_inverse12', 'LABEL_0', 'cased', 8),
          (tc_6, 'tc_6', 'NO_HATE', 'cased', 9),
          (tc_7, 'tc_7', 'NON_HATE', 'cased', 10),
          (sentiment_vader, 'sentiment_vader', 'Non-Offensive', 'cased', 11),
          (sentiment_textblob, 'sentiment_textblob', 'LABEL_0', 'cased', 12),
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

binary_models = [(tc_8, 'tc_8', 'no-hate-speech'),
                 (offensive_lexicon, 'offensive_lexicon', 'LABEL_0'),
                 (test.target_classifier, 'spacy', None)]

task_a_answers_array = np.array([0 if x < 1102 else 1 for x in range(3494)])
task_b_answers_array = np.array([0 if x < 551 else 1 for x in range(2392)])
oth_answers_array = np.array([1 if x < 430 else 0 for x in range(1290)])
ind_answers_array = np.array([1 if 429 < x < 860 else 0 for x in range(1290)])
grp_answers_array = np.array([1 if x > 859 else 0 for x in range(1290)])


def evaluate(
        test_tweets: list):

    uncased_test_tweets = [tweet.lower() for tweet in test_tweets]

    figure, axis = plt.subplots(2, 5)

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
                
        elif 'sentiment' in name:
            if 'vader' in name:
                classifier_score = [custom_sigmoid(sentiment_vader(tweet)) for tweet in test_tweets]
            else:
                classifier_score = [custom_sigmoid(sentiment_textblob(tweet)) for tweet in test_tweets]

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
        task_c_predictions_array = task_a_predictions_array[2204:]

        precision, recall, thresholds = precision_recall_curve(
            task_a_answers_array, task_a_predictions_array)
        if index > 19:
            axis[0, 0].plot(recall, precision, label=name, ls=':')
        elif index > 9:
            axis[0, 0].plot(recall, precision, label=name, ls='--')
        else:
            axis[0, 0].plot(recall, precision, label=name)
        axis[0, 0].set_title("Task A PRC")
        axis[0, 0].set_xlabel("Recall")
        axis[0, 0].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_a_answers_array, task_a_predictions_array)
        if index > 19:
            axis[1, 0].plot(fpr, recall, label=name, ls=':')
        elif index > 9:
            axis[1, 0].plot(fpr, recall, label=name, ls='--')
        else:
            axis[1, 0].plot(fpr, recall, label=name)
        axis[1, 0].set_title("Task A ROC")
        axis[1, 0].set_xlabel("False Positive Rate")
        axis[1, 0].set_ylabel("True Positive Rate")

        precision, recall, thresholds = precision_recall_curve(
            task_b_answers_array, task_b_predictions_array)
        if index > 19:
            axis[0, 1].plot(recall, precision, label=name, ls=':')
        elif index > 9:
            axis[0, 1].plot(recall, precision, label=name, ls='--')
        else:
            axis[0, 1].plot(recall, precision, label=name)
        axis[0, 1].set_title("Task B PRC")
        axis[0, 1].set_xlabel("Recall")
        axis[0, 1].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_b_answers_array, task_b_predictions_array)
        if index > 19:
            axis[1, 1].plot(fpr, recall, label=name, ls=':')
        elif index > 9:
            axis[1, 1].plot(fpr, recall, label=name, ls='--')
        else:
            axis[1, 1].plot(fpr, recall, label=name)
        axis[1, 1].set_title("Task B ROC")
        axis[1, 1].set_xlabel("False Positive Rate")
        axis[1, 1].set_ylabel("True Positive Rate")

        precision, recall, thresholds = precision_recall_curve(
            oth_answers_array, task_c_predictions_array)
        if index > 19:
            axis[0, 2].plot(recall, precision, label=name, ls=':')
        elif index > 9:
            axis[0, 2].plot(recall, precision, label=name, ls='--')
        else:
            axis[0, 2].plot(recall, precision, label=name)
        axis[0, 2].set_title("OTH PRC")
        axis[0, 2].set_xlabel("Recall")
        axis[0, 2].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            oth_answers_array, task_c_predictions_array)
        if index > 19:
            axis[1, 2].plot(fpr, recall, label=name, ls=':')
        elif index > 9:
            axis[1, 2].plot(fpr, recall, label=name, ls='--')
        else:
            axis[1, 2].plot(fpr, recall, label=name)
        axis[1, 2].set_title("OTH ROC")
        axis[1, 2].set_xlabel("False Positive Rate")
        axis[1, 2].set_ylabel("True Positive Rate")

        precision, recall, thresholds = precision_recall_curve(
            ind_answers_array, task_c_predictions_array)
        if index > 19:
            axis[0, 3].plot(recall, precision, label=name, ls=':')
        elif index > 9:
            axis[0, 3].plot(recall, precision, label=name, ls='--')
        else:
            axis[0, 3].plot(recall, precision, label=name)
        axis[0, 3].set_title("IND PRC")
        axis[0, 3].set_xlabel("Recall")
        axis[0, 3].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            ind_answers_array, task_c_predictions_array)
        if index > 19:
            axis[1, 3].plot(fpr, recall, label=name, ls=':')
        elif index > 9:
            axis[1, 3].plot(fpr, recall, label=name, ls='--')
        else:
            axis[1, 3].plot(fpr, recall, label=name)
        axis[1, 3].set_title("IND ROC")
        axis[1, 3].set_xlabel("False Positive Rate")
        axis[1, 3].set_ylabel("True Positive Rate")

        precision, recall, thresholds = precision_recall_curve(
            grp_answers_array, task_c_predictions_array)
        if index > 19:
            axis[0, 4].plot(recall, precision, label=name, ls=':')
        elif index > 9:
            axis[0, 4].plot(recall, precision, label=name, ls='--')
        else:
            axis[0, 4].plot(recall, precision, label=name)
        axis[0, 4].set_title("GRP PRC")
        axis[0, 4].set_xlabel("Recall")
        axis[0, 4].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            grp_answers_array, task_c_predictions_array)
        if index > 19:
            axis[1, 4].plot(fpr, recall, label=name, ls=':')
        elif index > 9:
            axis[1, 4].plot(fpr, recall, label=name, ls='--')
        else:
            axis[1, 4].plot(fpr, recall, label=name)
        axis[1, 4].set_title("GRP ROC")
        axis[1, 4].set_xlabel("False Positive Rate")
        axis[1, 4].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_binary(test_tweets: list):

    for classifier, name, wrong_label in models:
        if '8' in name:
            results = classifier(test_tweets)
            classifier_score = [0 if result['generated_text'] == wrong_label else 1 for result in results]

        elif 'offensive' in name:
            classifier_score = [1 if any(offensive_word in uncased_tweet for offensive_word in offensive_lexicon) else 0 for uncased_tweet in uncased_test_tweets]

        elif 'spacy' in name:
            classifier_score = np.zeros(length)
            results = classifier([tweet.lower() for tweet in test_tweets])
            for i in range(length):
                if i < 2634 and predictions[i] == 'OTH':
                    classifier_score[i] = 1
                elif i < 3064 and predictions[i] == 'IND':
                    classifier_score[i] = 1
                elif predictions[i] == 'GRP':
                    classifier_score[i] = 1

        predictions_array = np.array(classifier_score)

        true_positive_not = 0
        false_positive_not = 0

        true_positive_off = 0
        false_positive_off = 0

        true_positive_unt = 0
        false_positive_unt = 0

        true_positive_tin = 0
        false_positive_tin = 0

        true_positive_ind = 0
        false_positive_ind = 0

        true_positive_grp = 0
        false_positive_grp = 0

        true_positive_oth = 0
        false_positive_oth = 0

        for i in range(3494):
            if i < 1101:
                if predictions_array[i] == 0:
                    true_positive_not += 1
                else:
                    false_positive_off += 1

            else:
                if predictions_array[i]:
                    true_positive_off += 1
                else:
                    false_positive_not += 1

                if i < 1652:
                    if predictions_array[i] == 0:
                        true_positive_unt += 1
                    else:
                        false_positive_tin += 1

                else:
                    if predictions_array[i]:
                        true_positive_tin += 1
                    else:
                        false_positive_unt += 1

# confusion matrix






        metrics = [(true_positive_not, false_positive_not, 'NOT'),
                   (true_positive_off, false_positive_off, 'OFF'),
                   (true_positive_unt, false_positive_unt, 'UNT'),
                   (true_positive_tin, false_positive_tin, 'TIN'),
                   (true_positive_ind, false_positive_ind, 'IND'),
                   (true_positive_grp, false_positive_grp, 'GRP'),
                   (true_positive_oth, false_positive_oth, 'OTH')]

        for metric in metrics:
            pp = metric[0] + metric[1]
            precision = metric[0] / pp if pp > 0 else 0
            recall = metric[0] / category_frequency[metric[2]]
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


    # evaluate(
    #     test_tweets=filtered_tweets)

    evaluate_binary(
        test_tweets=filtered_tweets)
