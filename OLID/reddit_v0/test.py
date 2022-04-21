import spacy

nlp_trf = spacy.load('en_core_web_trf')

text = 'My first birthday was great. My 2. was even better.'
for i in nlp_trf(text).sents:
	print(i)