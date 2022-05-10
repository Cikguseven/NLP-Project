from os import listdir
from sklearn.metrics import precision_recall_curve, roc_curve
import comment_filter
import main_config
import matplotlib.pyplot as plt
import numpy as np
import spacy
import weak_signals

spacy.require_gpu()


def evaluate_model(
        models: list,
        test_comments: list,
        test_answers: list):

    task_a_answers_array = np.array(
        [1 if x[0] == 'OFF' else 0 for x in test_answers])
    task_b_answers_array = np.array(
        [1 if x[1] == 'TIN' else 0 for x in test_answers])

    uncased_test_comments = [comment.lower() for comment in test_comments]

    figure, axis = plt.subplots(2, 2)

    for model in models:

        print(model)

        if 'weak_signals_function' in model:
            task_a_predictions_array = weak_signals.model_aggregator(
                test_comments, uncased_test_comments, 'a')
            task_b_predictions_array = weak_signals.model_aggregator(
                test_comments, uncased_test_comments, 'b')

        else:
            nlp = spacy.load(main_config.model_directory +
                             model + '/model-best')

            if 'uncased' in model:
                docs = list(nlp.pipe(uncased_test_comments))
            else:
                docs = list(nlp.pipe(test_comments))

            task_a_predictions_array = np.array(
                [doc.cats['offensive'] for doc in docs])
            task_b_predictions_array = np.array(
                [doc.cats['targeted'] for doc in docs])

        precision, recall, thresholds = precision_recall_curve(
            task_a_answers_array, task_a_predictions_array)
        axis[0, 0].plot(recall, precision, label=model)
        axis[0, 0].set_title("Task A PRC")
        axis[0, 0].set_xlabel("Recall")
        axis[0, 0].set_ylabel("Precision")

        precision, recall, thresholds = precision_recall_curve(
            task_b_answers_array, task_b_predictions_array)
        axis[0, 1].plot(recall, precision, label=model)
        axis[0, 1].set_title("Task B PRC")
        axis[0, 1].set_xlabel("Recall")
        axis[0, 1].set_ylabel("Precision")

        fpr, recall, thresholds = roc_curve(
            task_a_answers_array, task_a_predictions_array)
        axis[1, 0].plot(fpr, recall, label=model)
        axis[1, 0].set_title("Task A ROC")
        axis[1, 0].set_xlabel("False Positive Rate")
        axis[1, 0].set_ylabel("True Positive Rate")

        fpr, recall, thresholds = roc_curve(
            task_b_answers_array, task_b_predictions_array)
        axis[1, 1].plot(fpr, recall, label=model)
        axis[1, 1].set_title("Task B ROC")
        axis[1, 1].set_xlabel("False Positive Rate")
        axis[1, 1].set_ylabel("True Positive Rate")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    custom_models = [f for f in listdir(
        main_config.model_directory)]

    specific_model = ['weak_signals_function']

    all_models = custom_models + specific_model

    # Import unique filtered comments for testing
    filtered_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=99,
        uncased=False,
        unique=False,
        input_file=main_config.hand_labelled_comments)

    comment_limit = 900

    evaluate_model(
        models=custom_models,
        test_comments=filtered_comments[comment_limit:],
        test_answers=main_config.answers[comment_limit:])
