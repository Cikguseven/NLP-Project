import main_config
import spacy
from os import listdir

# custom_models = [f for f in listdir(
#     main_config.NER_model_directory) if 'v' in f]

nlp = spacy.load('en_core_web_sm')
nlp_trf = spacy.load('en_core_web_trf')

sentence = '@USER was literally just talking about this lol all mass shootings like that have been set ups. it’s propaganda used to divide us on major issues like gun control and terrorism'

doc = nlp(sentence)

doc1 = nlp_trf(sentence)

for chunk in doc.noun_chunks:
	print(chunk)

print()
print()

for chunk in doc1.noun_chunks:
	print(chunk)

