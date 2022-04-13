import emoji
import spacy
import main_config
import time
from spacy.tokens import DocBin

start = time.time()


def preprocess(input_file: str):
    with open(input_file, encoding='utf-8') as f:
        preprocessed_tweets = [line.split('\t') for line in f]

        preprocessed_tweets.pop(0)

        for line in preprocessed_tweets:
            line[1] = line[1].replace('@USER ', '')
            line[1] = emoji.demojize(line[1], delimiters=(' ', ' '))
            line[4] = line[4].strip()

    return preprocessed_tweets


training_data = preprocess(main_config.training_tweets)
training_text = [item[1] for item in training_data]

spacy.require_gpu()

nlp = spacy.load('en_core_web_trf')
# nlp.add_pipe('textcat_multilabel')
# textcat = nlp.get_pipe('textcat_multilabel')

# labels = ['offensive', 'targeted', 'individual', 'group', 'other']

# for label in labels:
#     textcat.add_label(label)

db_training = DocBin()
db_validation = DocBin()

split = main_config.validation_split * len(training_text) // 100


for i in range(len(training_text)):
    categories = {
        'offensive': 0.0,
        'targeted': 0.0,
        'individual': 0.0,
        'group': 0.0,
        'other': 0.0
    }

    if training_data[i][2] == 'OFF':
        categories['offensive'] = 1.0

        if training_data[i][3] == 'TIN':
            categories['targeted'] = 1.0

            if training_data[i][4] == 'IND':
                categories['individual'] = 1.0

            elif training_data[i][4] == 'GRP':
                categories['group'] = 1.0

            else:
                categories['other'] = 1.0

    doc = nlp.make_doc(training_text[i])
    doc.cats = categories
    if i < split:
        db_validation.add(doc)
    else:
        db_training.add(doc)


db_training.to_disk(main_config.output_training)
db_validation.to_disk(main_config.output_vaildation)


end = time.time()
t = round(end - start, 2)

if t > 60:
    duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
else:
    duration = str(t) + 's'

print(f'Time taken: {duration}')
