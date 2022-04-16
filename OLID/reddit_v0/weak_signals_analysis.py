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


tc_pipe_0 = pipeline(task='text-classification', model="./pipelines/tc_pipe_0/", tokenizer="./pipelines/tc_pipe_0/", device=0)

tc_pipe_1 = pipeline(task='text-classification', model="./pipelines/tc_pipe_1/", tokenizer="./pipelines/tc_pipe_1/", device=0)

tc_pipe_2 = pipeline(task='text-classification', model="./pipelines/tc_pipe_2/", tokenizer="./pipelines/tc_pipe_2/", device=0)

tc_pipe_3 = pipeline(task='text-classification', model="./pipelines/tc_pipe_3/", tokenizer="./pipelines/tc_pipe_3/", device=0)

tc_pipe_4 = pipeline(task='text-classification', model="./pipelines/tc_pipe_4/", tokenizer="./pipelines/tc_pipe_4/", device=0)

tc_pipe_5 = pipeline(task='text-classification', model="./pipelines/tc_pipe_5/", tokenizer="./pipelines/tc_pipe_5/", device=0)

tc_pipe_6 = pipeline(task='text-classification', model="./pipelines/tc_pipe_6/", tokenizer="./pipelines/tc_pipe_6/", device=0)

tc_pipe_7 = pipeline(task='text-classification', model="./pipelines/tc_pipe_7/", tokenizer="./pipelines/tc_pipe_7/")

tc_pipe_8 = pipeline(task='text2text-generation', model="./pipelines/tc_pipe_8/", tokenizer="./pipelines/tc_pipe_8/")

tc_pipes = [tc_pipe_0, tc_pipe_1, tc_pipe_2, tc_pipe_3, tc_pipe_4, tc_pipe_5, tc_pipe_6, tc_pipe_7, tc_pipe_8]

sonar_model = Sonar()

detoxify_model = Detoxify('unbiased', device='cuda')

flair_model = TextClassifier.load('sentiment')

def test(sentences):
	for sentence in sentences:
		print(sentence)
		print()

		for pipe in tc_pipes:
			print(pipe(sentence))
			print()


		print('vader')
		print(custom_sigmoid(sentiment_vader(sentence)))
		print()

		print('textblob')
		print(custom_sigmoid(sentiment_textblob(sentence)))
		print()

		print('detoxify')
		print(detoxify_model.predict(sentence))
		print()

		print('sonar')
		print(sonar_model.ping(text=sentence)['classes'])
		print()


		print('flair')
		sentence = Sentence(sentence)
		flair_model.predict(sentence)
		print(sentence.labels[0].to_dict())
		print()
		print()


list_of_test = ['oh shit', 'fuck my life', 'bag of dicks']

test(list_of_test)