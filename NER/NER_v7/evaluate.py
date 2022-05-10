from os import listdir
import gold_labels
import main_config
import spacy
import time

spacy.require_gpu()

stopword_nlp = spacy.load("en_core_web_trf")
stopwords = stopword_nlp.Defaults.stop_words
stopwords.add('to be')
stopwords.add('think it')


def evaluate_model(
    models: list,
    answer_key: list,
    testing_lines: list,
    all_labels: str,
    wrong_labels: str):

    start = time.time()

    for model in models:

        # Load default spacy en_core_web models
        if model[0] == 'e':
            nlp = spacy.load(model)
        # Load custom update/rehearsal models
        elif model[0] == 'u' or model[0] == 'r':
            nlp = spacy.load(main_config.model_directory + model)
        # Load custom trained models
        else:
            nlp = spacy.load(main_config.model_directory + model + '/model-best')

        docs = list(nlp.pipe(testing_lines))

        true_positive = 0
        false_positive = 0

        with open(all_labels, 'w') as all_labels_file, open(wrong_labels, 'w') as wrong_labels_file:

            # all_labels_file.write(f'TEXT, START INDEX, END INDEX, ENTITY, SENTENCE INDEX\n')
            # wrong_labels_file.write(f'TEXT, START INDEX, END INDEX, ENTITY, SENTENCE INDEX\n')

            for i in range(len(testing_lines)):
                testing_lines_copy = testing_lines[i]

                for X in docs[i].ents:

                    # Filter stopwords
                    if X.text.lower() not in stopwords and main_config.label_dict[X.label_]:

                        new_label = [X.text, X.start_char, X.end_char, main_config.label_dict[X.label_]]

                        all_labels_file.write(f'{", ".join(map(str, new_label))}, {i + 1}\n')

                        if new_label in answer_key[i]:
                            true_positive += 1
                        else:
                            false_positive += 1
                            wrong_labels_file.write(f'{", ".join(map(str, new_label))}, {i + 1}\n')


        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / main_config.recursive_len(answer_key)
        f1 = 2 * precision * recall / (precision + recall)

        print(f'{model}, {precision}, {recall}, {f1}')


    end = time.time()
    t = round(end - start, 2)

    if t > 60:
        duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
    else:
        duration = str(t) + 's'

    print(f'Time taken: {duration}')

if __name__ == '__main__':
    specific_model = ['en_core_web_trf']

    custom_models = [f for f in listdir(main_config.model_directory) if 'v7' in f]

    spacy_models = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_core_web_trf']

    all_models = custom_models + spacy_models

    evaluate_model(
        models=all_models,
        answer_key=gold_labels.answer_key[:],
        testing_lines=main_config.testing_lines[:],
        all_labels=main_config.all_labels,
        wrong_labels=main_config.wrong_labels)
