import spacy
import main_config

spacy.require_gpu()

test_comments = ['fuck', 'I want to knife all women', 'the sun is yellow', 'why is this happening?']

nlp = spacy.load(main_config.model_directory + 'v0_weak_signals_25a_25b_lexicon15' + '/model-best')
docs = list(nlp.pipe(test_comments))

for doc in docs:
	print(doc.cats)
	print()
