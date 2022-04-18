from bad_words import offensive_lexicon
import weak_signals

offensive_lexicon_list = list(offensive_lexicon)[:]

scores = weak_signals.model_aggregator(comments=offensive_lexicon_list, uncased_comments=offensive_lexicon_list, task='b')

for i in range(len(offensive_lexicon_list)):
    print(f'{offensive_lexicon_list[i]} | {scores[i]}')
    
