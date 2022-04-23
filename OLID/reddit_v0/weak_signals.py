from detoxify import Detoxify
from flair.data import Sentence
from flair.models import TextClassifier
from hatesonar import Sonar
from math import exp
from os import listdir
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import main_config
import numpy as np
from bad_words import offensive_lexicon
import target_classifier


def custom_sigmoid(x):
    return 1 / (1 + exp(2.5 * x + 0.125))


def sentiment_vader(sentence):
    return SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']


def sentiment_textblob(sentence):
    return TextBlob(sentence).sentiment.polarity


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

tc_tuples = [(tc_0, 0, 'POSITIVE', 'cased', 1),
             (tc_1, 1, 'POSITIVE', 'uncased', 1),
             (tc_2, 2, 'LABEL_0', 'cased', 0.5),
             (tc_3, 3, 'LABEL_0', 'cased', 2),
             (tc_4, 4, 'Non-Offensive', 'cased', 2),
             (tc_5, 5, 'LABEL_0', 'cased', 2),
             (tc_6, 6, 'NO_HATE', 'cased', 1),
             (tc_7, 7, 'NON_HATE', 'cased', 1),
             (tc_8, 8, 'no-hate-speech', 'cased', 1)]

sonar_model = Sonar()

detoxify_model = Detoxify('unbiased', device='cuda')

flair_model = TextClassifier.load('sentiment')

detoxify_weight = 2
vader_weight = 1
textblob_weight = 1
sonar_weight = 1
lexicon_weight = 15
flair_weight = 1


def model_aggregator(comments: list,
                     uncased_comments: list):

    length = len(comments)

    task_a_score = np.zeros(length)
    task_a_weight = 0

    task_b_score = np.zeros(length)
    task_b_weight = 0

    # Combine all 3 tasks, port target classifier to this file, generate spacy and train

    if task == 'a':
        for classifier, index, wrong_label, case in tc_tuples:
            if case == 'uncased':
                results = classifier(uncased_comments)
            else:
                results = classifier(comments)

            if index < 8:
                classifier_score = [1 - result['score'] if result['label'] == wrong_label else result['score'] for result in results]

                if 2 < index < 6:
                    weight = 2
                else:
                    weight = 1

                task_a_score += np.array(classifier_score) * weight
                task_a_weight += weight

            else:
                classifier_score = [0 if result['generated_text']
                                    == wrong_label else 1 for result in results]

                task_a_score += np.array(classifier_score)
                task_a_weight += 1

        task_a_score += np.array(detoxify_model.predict(comments)['toxicity'])
        task_a_score += np.array(detoxify_model.predict(comments)['insult'])
        task_a_weight += detoxify_weight

        for i in range(length):
            vader_score = custom_sigmoid(sentiment_vader(comments[i]))

            textblob_score = custom_sigmoid(sentiment_textblob(comments[i]))

            sonar_score = 1 - \
                sonar_model.ping(text=comments[i])['classes'][2]['confidence']

            lexicon_score = 1 if any(
                offensive_word in uncased_comments[i] for offensive_word in offensive_lexicon) else 0

            sentence = Sentence(comments[i])
            flair_model.predict(sentence)
            flair_result = sentence.labels[0].to_dict()
            flair_score = 1 - \
                flair_result['confidence'] if flair_result['value'] == 'POSITIVE' else flair_result['confidence']

            task_a_score[i] += (vader_score * vader_weight + textblob_score * textblob_weight + sonar_score * sonar_weight + lexicon_score * lexicon_weight + flair_score * flair_weight)
        
        task_a_weight += vader_weight + textblob_weight + sonar_weight + lexicon_weight + flair_weight

        task_a_score /= task_a_weight

    elif task == 'b':
        for classifier, index, wrong_label, case, weight in tc_tuples:
            if index == 0 or index > 4:
                results = classifier(comments)

                if index == 5:
                    task_b_score += np.array([1 - result['score'] if result['label'] in (
                        'LABEL_0', 'LABEL_1') else result['score'] for result in results]) * weight
                    task_b_weight += weight

                elif index < 8:
                    task_b_score += np.array([1 - result['score'] if result['label'] == wrong_label else result['score'] for result in results]) * weight
                    task_b_weight += weight

                else:
                    task_b_score += np.array([0 if result['generated_text']
                                               == wrong_label else 1 for result in results]) * weight
                    task_b_weight += weight

        task_b_score += np.array(detoxify_model.predict(comments)['insult']) * detoxify_weight
        task_b_weight += detoxify_weight

        for i in range(length):
            sonar_output = sonar_model.ping(text=comments[i])['classes']
            sonar_score = sonar_output[0]['confidence'] / (1 - sonar_output[2]['confidence'])

            sentence = Sentence(comments[i])
            flair_model.predict(sentence)
            flair_result = sentence.labels[0].to_dict()
            flair_score = 1 - \
                flair_result['confidence'] if flair_result['value'] == 'POSITIVE' else flair_result['confidence']

            task_b_score[i] += sonar_score * sonar_weight + flair_score * flair_weight
        
        task_b_weight += sonar_weight + flair_weight

        task_b_score /= task_b_weight

    elif task == 'c':
        overall_score = target_classifier.weak_classifier(comments)

    return overall_score


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

    filtered_uncased_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=True,
        length_min=0,
        length_max=99,
        uncased=True,
        unique=False,
        input_file=main_config.hand_labelled_comments)

    print(
        model_aggregator(
            comments=filtered_cased_comments,
            uncased_comments=filtered_uncased_comments,
            task='b'))
