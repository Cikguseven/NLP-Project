import comment_filter
import main_config
import spacy
from os import listdir

spacy.require_gpu()

with open('hwz.csv', encoding='utf-8') as f:
	hwz_comments = [line.split(',', 2)[2].strip() for line in f]

hwz_comments.pop(0)

# Import unique filtered comments for testing
filtered_hwz_comments = comment_filter.c_filter(
    shuffle=False,
    remove_username=False,
    remove_commas=False,
    length_min=10,
    length_max=9999,
    uncased=False,
    unique=True,
    input_list=hwz_comments)

# Import unique filtered comments for testing
filtered_gab_comments = comment_filter.c_filter(
    shuffle=False,
    remove_username=False,
    remove_commas=False,
    length_min=10,
    length_max=9999,
    uncased=False,
    unique=True,
    input_file='gab_comments.txt')
    
models = [f for f in listdir(main_config.model_directory) if 'b_' in f]

offensive_threshold = 0.5
targeted_threshold = 0.5

for model in models:

    print(model)

    nlp = spacy.load(main_config.model_directory +
                        model + '/model-best')

    data = [('hwz', filtered_hwz_comments), ('gab', filtered_gab_comments)]

    for d in data:
        docs = list(nlp.pipe(d[1]))

        off = 0
        tin = 0
        ind = 0
        grp = 0
        oth = 0

        for doc in docs:
            result = doc.cats

            if result["offensive"] > offensive_threshold:
                off += 1

                if result["targeted"] > targeted_threshold:
                    tin += 1

                    result.pop('offensive')
                    result.pop('targeted')
                    prediction = max(result, key=result.get)

                    if prediction == 'individual':
                        ind += 1

                    elif prediction == 'group':
                        grp += 1

                    elif prediction == 'other':
                        oth += 1

        print(f'{d[0]}: {off} | {tin} | {ind} | {grp} | {oth}')

