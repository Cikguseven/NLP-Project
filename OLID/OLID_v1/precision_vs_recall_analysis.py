from os import listdir
import main_config
import spacy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import time

spacy.require_gpu()


def evaluate_model(
        models: list,
        cased_test_tweets: list,
        test_answers: list):

    task_a_answers_array = np.array([1 if x == 'OFF' else 0 for x in test_answers[:860]])
    task_b_answers_array = np.array([1 if x == 'TIN' else 0 for x in test_answers[860:1100]])

    for model in models:

        start = time.time()

        print(model)

        if 'uncased' in model:
            all_tweets = [tweet.lower() for tweet in cased_test_tweets]
        else:
            all_tweets = cased_test_tweets

        nlp = spacy.load(main_config.model_directory + model + '/model-best')
        docs = list(nlp.pipe(all_tweets))

        task_a_predictions_array = np.array([docs[i].cats['offensive'] for i in range(860)])
        task_b_predictions_array = np.array([docs[i].cats['targeted'] for i in range(860, 1100)])

        # task_a_precision, task_a_recall, thresholds = precision_recall_curve(
        #     task_a_answers_array, task_a_predictions_array)

        # plt.plot(task_a_recall, task_a_precision, label = model)

        task_b_precision, task_b_recall, thresholds = precision_recall_curve(
            task_b_answers_array, task_b_predictions_array)

        plt.plot(task_b_recall, task_b_precision, label = model)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    custom_models = [f for f in listdir(
        main_config.model_directory)]

    evaluate_model(
        models=custom_models,
        cased_test_tweets=main_config.preprocess(
            tweets=main_config.test_tweets_getter(), uncased=False),
        test_answers=main_config.answers_getter())
