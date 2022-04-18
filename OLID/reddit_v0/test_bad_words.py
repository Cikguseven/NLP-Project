from bad_words import offensive_lexicon
import weak_signals

scores = weak_signals.model_aggregator(comments=offensive_lexicon, uncased_comments=offensive_lexicon, task='b')

for i in range(len(offensive_lexicon)):
    print(f'{offensive_lexicon[i]} | {scores[i]}')
    
