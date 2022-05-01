from detoxify import Detoxify
from flair.data import Sentence
from flair.models import TextClassifier
from hatesonar import Sonar
from math import exp
from os import listdir
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import bad_words
import main_config
import numpy as np
import re
import target_classifier


def custom_sigmoid(x):
    return 1 / (1 + exp(2.5 * x + 0.125))


def vader(sentence):
    return SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']


def textblob(sentence):
    return TextBlob(sentence).sentiment.polarity


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

sonar_model = Sonar()

detoxify_model = Detoxify('unbiased', device='cuda')

flair_model = TextClassifier.load('sentiment')

models = [(tc_1, 'tc_1', 'POSITIVE', 'uncased', 1),
          (tc_2, 'tc_2', 'LABEL_0', 'cased', 0.5),
          (tc_3, 'tc_3', 'LABEL_0', 'cased', 2),
          (tc_4, 'tc_4', 'Non-Offensive', 'cased', 2),
          (tc_5, 'tc_5', 'LABEL_0', 'cased', 2),
          (tc_6, 'tc_6', 'NO_HATE', 'cased', 1),
          (tc_7, 'tc_7', 'NON_HATE', 'cased', 1),
          (tc_8, 'tc_8', 'POSITIVE', 'cased', 1),
          (bad_words.offensive_lexicon, 'lexicon', None, 'uncased', 1),
          (target_classifier.weak_classifier, 'target_classifier', None, 'cased', 1),
          (vader, 'vader', None, 'cased', 1),
          (textblob, 'textblob', None, 'cased', 1),
          (sonar_model, 'sonar', None, 'cased', 1),
          (detoxify_model, 'detoxify_toxicity', 'toxicity', 'cased', 1),
          (detoxify_model, 'detoxify_insult', 'insult', 'cased', 1),
          (flair_model, 'flair', None, 'cased', 1)]


def model_aggregator(comments: list):

    length = len(comments)

    uncased_comments = [comment.lower() for comment in comments]

    task_a_score = np.zeros(length)
    task_a_weight = 0

    task_b_score = np.zeros(length)
    task_b_weight = 0

    for classifier, name, keyword, case, weight in models:

        print(name)

        classifier_array_b = None

        if 'tc_' in name:
            if case == 'uncased':
                results = classifier(uncased_comments)
            else:
                results = classifier(comments)

            classifier_array_a = [1 - result['score'] if result['label'] == keyword else result['score'] for result in results]

            if int(name[-1]) > 5:
                classifier_array_b = classifier_array_a
            elif '5' in name:
                classifier_array_b = [1 - result['score'] if result['label'] in ('LABEL_0', 'LABEL_1') else result['score'] for result in results]

        elif 'lexicon' in name:
            classifier_array_a = np.zeros(length)
            for index, uncased_comment in enumerate(uncased_comments):
                for offensive_word in bad_words.offensive_lexicon:
                    if re.search(r'(?<![^\W_])' + offensive_word + r'(?![^\W_])', uncased_comment):
                        classifier_array_a[index] = 1
                        break

        elif 'target' in name:
            task_c_score = classifier(comments)

        elif 'vader' in name:
            classifier_array_a = [custom_sigmoid(vader(comment)) for comment in comments]

        elif 'textblob' in name:
            classifier_array_a = [custom_sigmoid(textblob(comment)) for comment in comments]

        elif 'sonar' in name:
            sonar_results = [classifier.ping(text=comment)['classes'] for comment in comments]
            classifier_array_a = np.array([1 - result[2]['confidence'] for result in sonar_results])
            classifier_array_b = np.array([result[0]['confidence'] for result in sonar_results]) / classifier_array_a

        elif 'detoxify' in name:
            classifier_array_a = [detoxify_model.predict(comment)[keyword] for comment in comments]

            if keyword == 'insult':
                classifier_array_b = classifier_array_a

        elif 'flair' in name:
            classifier_array_a = []
            
            for comment in comments:
                flair_sentence = Sentence(comment)
                classifier.predict(flair_sentence)
                flair_result = flair_sentence.labels[0].to_dict()
                flair_score = 1 - flair_result['confidence'] if flair_result['value'] == 'POSITIVE' else flair_result['confidence']
                classifier_array_a.append(flair_score)

            classifier_array_b = classifier_array_a

        task_a_score += np.array(classifier_array_a) * weight
        task_a_weight += weight

        if classifier_array_b is not None:
            task_b_score += np.array(classifier_array_b) * weight
            task_b_weight += weight

    task_a_score /= task_a_weight
    task_b_score /= task_a_weight

    return task_a_score, task_b_score, task_c_score


if __name__ == '__main__':
    import comment_filter

    # Import unique filtered comments for testing
    filtered_cased_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=True,
        length_min=0,
        length_max=99,
        uncased=False,
        unique=False,
        input_file=main_config.hand_labelled_comments)

    print(
        model_aggregator(
            comments=filtered_cased_comments))
