import main_config
from spacy.tokens import DocBin
import spacy
import time

start = time.time()

spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')

training_data = main_config.training_tweets_getter()

training_text = main_config.preprocess(
    tweets=[item[1] for item in training_data], uncased=False)

for i in range(len(training_text)):
    print(f'{training_text[i]} | {training_data[i][2]} | {training_data[i][3]} | {training_data[i][4]}')

# # print(main_config.k_most_frequent_words(data=training_text, k=300))

# db_training = DocBin()
# db_validation = DocBin()

# split = main_config.validation_split * len(training_text) // 100

# for i in range(len(training_text)):
#     categories = {
#         'offensive': 0.0,
#         'targeted': 0.0,
#         'individual': 0.0,
#         'group': 0.0,
#         'other': 0.0
#     }

#     if training_data[i][2] == 'OFF':
#         categories['offensive'] = 1.0

#         if training_data[i][3] == 'TIN':
#             categories['targeted'] = 1.0

#             if training_data[i][4] == 'IND':
#                 categories['individual'] = 1.0

#             elif training_data[i][4] == 'GRP':
#                 categories['group'] = 1.0

#             else:
#                 categories['other'] = 1.0

#     doc = nlp.make_doc(training_text[i])
#     doc.cats = categories

#     if i < split:
#         db_validation.add(doc)

#     else:
#         db_training.add(doc)

# db_training.to_disk(main_config.output_training)
# db_validation.to_disk(main_config.output_vaildation)

# end = time.time()
# t = round(end - start, 2)

# if t > 60:
#     duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
# else:
#     duration = str(t) + 's'

# print(f'Time taken: {duration}')
