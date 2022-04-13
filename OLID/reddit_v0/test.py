import main_config
import spacy

stopword_nlp = spacy.load("en_core_web_trf")
stopwords = stopword_nlp.Defaults.stop_words

with open(main_config.hand_labelled_comments) as f:
    lines = [next(f).strip()
             for x in range(1000) if main_config.answers[x][0] == 'OFF']

for word in main_config.k_most_frequent_words(lines, 300):
    if word[0].lower() not in stopwords:
        print(word)

# # Check if labels correspond to sentence
# for x in range(1000):
#     if main_config.answers[x][0] == 'OFF':
#         print(lines[x])
#     else:
#         print()

