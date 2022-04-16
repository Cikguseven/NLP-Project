import bad_words
import weak_signals

for word in bad_words.offensive_words:
    if weak_signals.model_aggregator(
            comments=[word],
            uncased_comments=[word],
            task='a',
            device='cpu') < 0.5:
        print(word)
