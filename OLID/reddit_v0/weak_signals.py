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
import spacy


def custom_sigmoid(x):
    return 1 / (1 + exp(2.5 * x + 0.125))


def sentiment_vader(sentence):
    return SentimentIntensityAnalyzer().polarity_scores(sentence)['compound']


def sentiment_textblob(sentence):
    return TextBlob(sentence).sentiment.polarity


# Hate speech text classifier models from Hugging Face

# GPU
tc_pipe_0 = pipeline(task='text-classification', model="./pipelines/tc_pipe_0/", tokenizer="./pipelines/tc_pipe_0/", device=0)

tc_pipe_1 = pipeline(task='text-classification', model="./pipelines/tc_pipe_1/", tokenizer="./pipelines/tc_pipe_1/", device=0)

tc_pipe_2 = pipeline(task='text-classification', model="./pipelines/tc_pipe_2/", tokenizer="./pipelines/tc_pipe_2/", device=0)

tc_pipe_3 = pipeline(task='text-classification', model="./pipelines/tc_pipe_3/", tokenizer="./pipelines/tc_pipe_3/", device=0)

tc_pipe_4 = pipeline(task='text-classification', model="./pipelines/tc_pipe_4/", tokenizer="./pipelines/tc_pipe_4/", device=0)

tc_pipe_5 = pipeline(task='text-classification', model="./pipelines/tc_pipe_5/", tokenizer="./pipelines/tc_pipe_5/", device=0)

tc_pipe_6 = pipeline(task='text-classification', model="./pipelines/tc_pipe_6/", tokenizer="./pipelines/tc_pipe_6/", device=0)

tc_pipe_7 = pipeline(task='text-classification', model="./pipelines/tc_pipe_7/", tokenizer="./pipelines/tc_pipe_7/", device=0)

tc_pipe_8 = pipeline(task='text2text-generation', model="./pipelines/tc_pipe_8/", tokenizer="./pipelines/tc_pipe_8/", device=0)

tc_tuples = [(tc_pipe_0, 0, 'POSITIVE', 'cased'),
             (tc_pipe_1, 1, 'POSITIVE', 'uncased'),
             (tc_pipe_2, 2, 'LABEL_0', 'cased'),
             (tc_pipe_3, 3, 'LABEL_0', 'cased'),
             (tc_pipe_4, 4, 'Non-Offensive', 'cased'),
             (tc_pipe_5, 5, 'LABEL_0', 'cased'),
             (tc_pipe_6, 6, 'NO_HATE', 'cased'),
             (tc_pipe_7, 7, 'NON_HATE', 'cased'),
             (tc_pipe_8, 8, 'no-hate-speech', 'cased')]

sonar_model = Sonar()

detoxify_model = Detoxify('unbiased', device='cuda')

flair_model = TextClassifier.load('sentiment')

spacy.require_gpu()
spacy_models = [f for f in listdir(main_config.model_directory)]


def model_aggregator(comments: list,
                     uncased_comments: list,
                     task: str):

    length = len(comments)

    overall_score = np.zeros(length)

    if task == 'a':
        for classifier, index, wrong_label, case in tc_tuples:
            if case == 'uncased':
                results = classifier(uncased_comments)
            else:
                results = classifier(comments)

            if index < 8:
                classifier_score = [1 - result['score'] if result['label'] == wrong_label else result['score'] for result in results]

                if index < 6:
                    overall_score += np.array(classifier_score)

                else:
                    overall_score += np.array(classifier_score) * 0.5
            
            else:
                classifier_score = [0 if result['generated_text'] == wrong_label else 1 for result in results]
                overall_score += np.array(classifier_score)

        for model in spacy_models:
            nlp = spacy.load(main_config.model_directory +
                             model + '/model-best')

            if 'uncased' in model:
                docs = list(nlp.pipe(uncased_comments))
            else:
                docs = list(nlp.pipe(comments))

            overall_score += np.array(
                [doc.cats['offensive'] for doc in docs]) * 4

        for i in range(length):
            vader_score = custom_sigmoid(sentiment_vader(comments[i])) 
            textblob_score = custom_sigmoid(sentiment_textblob(comments[i]))

            detoxify_score = detoxify_model.predict(comments[i])['toxicity']

            sonar_score = 1 - \
                sonar_model.ping(text=comments[i])['classes'][2]['confidence']

            lexicon_score = 1 if any(offensive_word in comments[i] for offensive_word in offensive_lexicon) else 0

            sentence = Sentence(comments[i])
            flair_model.predict(sentence)
            flair_result = sentence.labels[0].to_dict()
            flair_score = 1 - flair_result['confidence'] if flair_result['value'] == 'POSITIVE' else flair_result['confidence']
            
            overall_score[i] += (vader_score + textblob_score * 0.5 + detoxify_score + sonar_score * 0.5 + lexicon_score * 5.5 + flair_score * 0.5)

        overall_score /= 25

        return overall_score

    elif task == 'b':
        for classifier, index, wrong_label, case in tc_tuples[5:]:
            results = classifier(comments)

            if index == 5:
                overall_score += np.array([1 - result['score'] if result['label'] in ('LABEL_0', 'LABEL_1') else result['score'] for result in results])

            elif index < 8:
                overall_score += np.array([1 - result['score'] if result['label'] == wrong_label else result['score'] for result in results])
            
            else:
                overall_score += np.array([0 if result['generated_text'] == wrong_label else 1 for result in results])

        for model in spacy_models:
            nlp = spacy.load(main_config.model_directory +
                             model + '/model-best')

            if 'uncased' in model:
                docs = list(nlp.pipe(uncased_comments))
            else:
                docs = list(nlp.pipe(comments))

            overall_score += np.array(
                [doc.cats['targeted'] for doc in docs]) * 4

        for i in range(length):
            detoxify_score = detoxify_model.predict(comments[i])['insult']

            sonar_output = sonar_model.ping(text=comments[i])['classes']
            sonar_score = sonar_output[0]['confidence'] / (1 - \
                sonar_output[2]['confidence'])

            overall_score[i] += detoxify_score + sonar_score

        overall_score /= 14

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