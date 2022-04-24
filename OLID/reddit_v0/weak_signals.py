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

tc_9 = pipeline(task='text2text-generation', model="./pipelines/tc_9/",
                tokenizer="./pipelines/tc_9/", device=0)

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
          (tc_9, 'tc_9', 'no-hate-speech', 'cased', 1),
          (offensive_lexicon, 'lexicon', None, 'uncased', 15),
          (target_classifier.weak_classifier, 'spacy', None, 'cased', 1),
          (vader, 'sentiment_vader', None, 'cased', 1),
          (textblob, 'sentiment_textblob', None, 'cased', 1),
          (sonar, 'sonar', None, 'cased', 1),
          (detoxify, 'detoxify_toxicity', 'toxicity', 'cased', 1),
          (detoxify, 'detoxify_insult', 'insult', 'cased', 1),
          (flair, 'flair', None, 'cased', 1)]


def model_aggregator(comments: list,
                     uncased_comments: list):

    length = len(comments)

    task_a_score = np.zeros(length)
    task_a_weight = 0

    task_b_score = np.zeros(length)
    task_b_weight = 0

    # Combine all 3 tasks, port target classifier to this file, generate spacy and train
    for classifier, name, keyword, case, weight in models:

        print(name)

        classifier_array_b = None

        if 'tc_' in name:
            if case == 'uncased':
                results = classifier(uncased_comments)
            else:
                results = classifier(comments)

            if '9' in name:
                classifier_array_a = [0 if result['generated_text'] == keyword else 1 for result in results]
            else:
                classifier_array_a = [1 - result['score'] if result['label'] == keyword else result['score'] for result in results]

            if int(name[0]) > 5:
                classifier_array_b = classifier_array_a
            elif '5' in name:
                classifier_array_b = [1 - result['score'] if result['label'] in ('LABEL_0', 'LABEL_1') else result['score'] for result in results]
        
        elif 'vader' in name:
            classifier_array_a = [custom_sigmoid(vader(tweet)) for tweet in test_tweets]

        elif 'textblob' in name:
            classifier_array_a = [custom_sigmoid(textblob(tweet)) for tweet in test_tweets]

        elif 'detoxify' in name:
            classifier_array_a = detoxify_model.predict(comments)[keyword]

            if keyword == 'insult':
                classifier_array_b = classifier_array_a

        task_a_score += np.array(classifier_array_a) * weight
        task_a_weight += weight

        if classifier_array_b is not None:
            task_b_score += np.array(classifier_array_b) * weight
            task_b_weight += weight

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
        for classifier, index, keyword, case, weight in tc_tuples:

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
