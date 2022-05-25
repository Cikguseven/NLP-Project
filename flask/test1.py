import flask_config
import spacy

# stopword_nlp = spacy.load("en_core_web_trf")
# stopwords = stopword_nlp.Defaults.stop_words
# stopwords.add('to be')

spacy.require_gpu()

def pipeline(input_sentence: str):
    # displaCy visualizer for NER
    ner_nlp = spacy.load(flask_config.ner_model)

    stopwords = ner_nlp.Defaults.stop_words
    stopwords.add('to be')

    ner_doc = ner_nlp(input_sentence)
    ner_doc.ents = tuple(x for x in ner_doc.ents if x.text.lower() not in stopwords)

    for x in ner_doc.ents:
        print(x)

sentence = 'Obama for you to be so dumb, you need to be think it right.'

pipeline(sentence)


