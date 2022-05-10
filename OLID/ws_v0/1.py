import spacy
import comment_filter
import main_config

spacy.require_gpu()

filtered_comments = comment_filter.c_filter(
        shuffle=True,
        remove_username=True,
        remove_commas=False,
        length_min=10,
        length_max=999,
        uncased=False,
        unique=True,
        edmw=False,
        input_file='redditsg_raw.txt')

nlp = spacy.load(main_config.model_directory + 'ws_v1_50a_10b_lexicon10_tc9removed' + '/model-best')

docs = list(nlp.pipe(filtered_comments))

offensive_threshold = 0.7
targeted_threshold = 0.2

for i in range(10000):
    doc = docs[i]
    is_off = 'NOT'
    is_tin = 'NULL'
    target = 'NULL'

    result = doc.cats

    if result["offensive"] > offensive_threshold:
        is_off = 'OFF'

        if result["targeted"] > targeted_threshold:
            is_tin = 'TIN'

            result.pop('offensive')
            result.pop('targeted')
            prediction = max(result, key=result.get)

            if prediction == 'individual':
                target = 'IND'
            elif prediction == 'group':
                target = 'GRP'
            else:
                target = 'OTH'

        else:
            is_tin = 'UNT'
            print(f'{doc.text} | OFF | UNT | NULL')

    #  print(f'{doc.text} | {is_off} | {is_tin} | {target}')
