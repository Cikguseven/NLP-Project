import spacy
import comment_filter
import main_config

spacy.require_gpu()

test_sentences = [
'fuck',
'shit',
'god damn it',
'you are not funny',
'they are being assholes',
'damn girl',
'cunt',
'cock',
'nigger',
'I want to punch you',
'I want to knife women',
'I am going to bomb that building',
'why is the sun black?',
'why am I so bad?',
'how is this possible?',
'let us meet at 3pm tomorrow',
'you dont cb',
'pundei',
'oranges are tasty'
]

model = 'wk14_ws_v1_60a_10b_lexicon1_tc9removed_13240'

is_edmw = False

if 'reddit' in model or 'hwz' in model:
    is_edmw = True

filtered_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=999,
        uncased=False,
        unique=False,
        edmw=is_edmw,
        input_list=test_sentences)

nlp = spacy.load(main_config.model_directory + model + '/model-best')

docs = list(nlp.pipe(filtered_comments))

offensive_threshold = 0.01
targeted_threshold = 0.95

for doc in docs:
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

    print(f'{doc.text} | {is_off} | {is_tin} | {target}')
