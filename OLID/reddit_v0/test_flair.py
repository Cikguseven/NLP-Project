from flair.models import TextClassifier
from flair.models import TARSClassifier
from flair.data import Sentence

# classifier.predict(text)
# value = text.labels[0].to_dict()['value']
#     if value == 'POSITIVE':
#         result = text.to_dict()['labels'][0]['confidence']
#     else:
#         result = -(text.to_dict()['labels'][0]['confidence'])
#     return round(result, 3)

# load tagger
classifier = TextClassifier.load('sentiment')

# make example sentence
sentence = Sentence("the tree is big")

# call predict
classifier.predict(sentence)

# check prediction
print(sentence.labels[0].to_dict()['value'])
print(sentence.labels[0].to_dict()['confidence'])
