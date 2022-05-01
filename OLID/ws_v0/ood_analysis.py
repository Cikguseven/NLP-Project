import comment_filter
import main_config
import spacy
from os import listdir

spacy.require_gpu()

with open('hwz.csv', encoding='utf-8') as f:
	hwz_comments = [line.split(',', 2)[2].strip() for line in f]

hwz_comments.pop(0)

offensive_threshold = 0.5
targeted_threshold = 0.5

# Import unique filtered comments for testing
filtered_hwz_comments = comment_filter.c_filter(
    shuffle=True,
    remove_username=False,
    remove_commas=False,
    length_min=10,
    length_max=99,
    uncased=False,
    unique=True,
    'edmw_mode',
    input_list=hwz_comments)

nlp_hwz = spacy.load(main_config.model_directory + 'ws_v1_50a_10b_lexicon10_tc9removed' + '/model-best')

docs_hwz = list(nlp_hwz.pipe(filtered_hwz_comments))

# # Import unique filtered comments for testing
# filtered_gab_comments = comment_filter.c_filter(
#     shuffle=True,
#     remove_username=False,
#     remove_commas=False,
#     length_min=10,
#     length_max=99,
#     uncased=False,
#     unique=True,
#     input_file='gab_comments.txt')

# nlp_gab = spacy.load(main_config.model_directory + 'ws_v1_30a_10b_lexicon1_tc9removed' + '/model-best')

# docs_gab = list(nlp_gab.pipe(filtered_gab_comments))

for doc in docs_hwz:
    result = doc.cats

    off = "NOT"
    tin = "NULL"
    c = "NULL"

    if result["offensive"] > offensive_threshold:
        off = "OFF"

        if result["targeted"] > targeted_threshold:
            tin = "TIN"

            result.pop('offensive')
            result.pop('targeted')
            prediction = max(result, key=result.get)

            if prediction == 'individual':
                c = "IND"

            elif prediction == 'group':
                c = "GRP"

            else:
                c = "OTH"

        else:
            tin = "UNT"

    print(f'{doc.text} | {off} | {tin} | {c}')
