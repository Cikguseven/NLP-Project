from detoxify import Detoxify
from flair.data import Sentence
from flair.models import TextClassifier
from hatesonar import Sonar
from math import exp
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import main_config
import numpy as np


def custom_sigmoid(x):
    return 1 / (1 + exp(2.5 * x + 0.125))


def sentiment_vader(sentence):
    return SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']


def sentiment_textblob(sentence):
    return TextBlob(sentence).sentiment.polarity


# Hate speech text classifier models from Hugging Face

tc_pipe_0 = pipeline(task='text-classification', model="./pipelines/tc_pipe_0/", tokenizer="./pipelines/tc_pipe_0/", device=0)

tc_pipe_1 = pipeline(task='text-classification', model="./pipelines/tc_pipe_1/", tokenizer="./pipelines/tc_pipe_1/", device=0)

tc_pipe_2 = pipeline(task='text-classification', model="./pipelines/tc_pipe_2/", tokenizer="./pipelines/tc_pipe_2/", device=0)

tc_pipe_3 = pipeline(task='text-classification', model="./pipelines/tc_pipe_3/", tokenizer="./pipelines/tc_pipe_3/", device=0)

tc_pipe_4 = pipeline(task='text-classification', model="./pipelines/tc_pipe_4/", tokenizer="./pipelines/tc_pipe_4/", device=0)

tc_pipe_5 = pipeline(task='text-classification', model="./pipelines/tc_pipe_5/", tokenizer="./pipelines/tc_pipe_5/", device=0)

tc_pipe_6 = pipeline(task='text-classification', model="./pipelines/tc_pipe_6/", tokenizer="./pipelines/tc_pipe_6/", device=0)


tc_tuples = [(tc_pipe_0, 'NON_HATE', 'cased'),
             (tc_pipe_1, 'POSITIVE', 'uncased'),
             (tc_pipe_2, 'LABEL_0', 'cased'),
             (tc_pipe_3, 'LABEL_0', 'cased'),
             (tc_pipe_4, 'Non-Offensive', 'cased'),
             (tc_pipe_5, 'POSITIVE', 'cased'),
             (tc_pipe_6, 'LABEL_0', 'cased')]


sonar_model = Sonar()

detoxify_model = Detoxify('unbiased', device='cuda')

flair_model = TextClassifier.load('sentiment')


def model_aggregator(comments: list,
                     uncased_comments: list):

    length = len(comments)

    weighted_average_score = np.zeros(length)

    for classifier, wrong_label, case in tc_tuples:
        if case == 'uncased':
            results = classifier(uncased_comments)
        else:
            results = classifier(comments)

        single_score = [1 - result['score'] if result['label'] ==
                        wrong_label else result['score'] for result in results]

        weighted_average_score += np.array(single_score)

    torch.cuda.empty_cache() 

    detoxify_score = detoxify_model.predict(comments)['toxicity']
    weighted_average_score += np.array(detoxify_score) * 0.5

    for i in range(length):
        vader_score = custom_sigmoid(sentiment_vader(comments[i])) 
        textblob_score = custom_sigmoid(sentiment_textblob(comments[i]))
        sonar_score = 1 - \
            sonar_model.ping(text=comments[i])['classes'][2]['confidence']

        sentence = Sentence(comments[i])
        flair_model.predict(sentence)
        flair_result = sentence.labels[0].to_dict()
        flair_score = 1 - flair_result['confidence'] if flair_result['value'] == 'POSITIVE' else flair_result['confidence']
        
        weighted_average_score[i] += (vader_score + textblob_score +
                    sonar_score + flair_score) * 0.5

    weighted_average_score /= 9.5

    return weighted_average_score


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
            uncased_comments=filtered_uncased_comments))
