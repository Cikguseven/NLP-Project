from os import listdir
import main_config
import spacy
import time
import numpy as np

spacy.require_gpu()


def evaluate_model(models: list):

    with open(main_config.testing_comments, encoding='utf-8') as f:
        comments = [c.strip() for c in f]

    for model in models:

        start = time.time()

        print(model)

        nlp = spacy.load(main_config.model_directory + model + '/model-best')
        docs = list(nlp.pipe(comments))

        a_counter = 0
        b_counter = 0
        c_ind = 0
        c_grp = 0
        c_oth = 0

        for doc in docs:
            result = doc.cats

            task_a = 'NOT'
            task_b = 'NULL'
            task_c = 'NULL'

            if result['offensive'] > 0.4:
                task_a = 'OFF'
                a_counter += 1

                if result['targeted'] > 0.4:
                    task_b = 'TIN'
                    b_counter += 1

                    result.pop('offensive')
                    result.pop('targeted')
                    c_prediction = max(result, key=result.get)
                    
                    if c_prediction == 'individual':
                        task_c = 'IND'
                        c_ind += 1

                    elif c_prediction == 'group':
                        task_c = 'GRP'
                        c_grp += 1

                    else:
                        task_c = 'OTH'
                        c_oth += 1

            print(f'{doc.text} | {task_a} | {task_b} | {task_c}')

        print(a_counter)
        print(b_counter)
        print(c_ind)
        print(c_grp)
        print(c_grp)
        print()   

            
        end = time.time()
        t = round(end - start, 2)

        if t > 60:
            duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
        else:
            duration = str(t) + 's'

        print(f'Time taken: {duration}')
        print()


if __name__ == '__main__':
    custom_models = [f for f in listdir(
        main_config.model_directory) if 'v1_cased_no' in f]

    evaluate_model(
        models=custom_models)
