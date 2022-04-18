import main_config
import weak_signals

with open(main_config.hand_labelled_comments) as f:
	comments = [c.strip() for c in f]

offensive_comments = []

for i in range(len(comments)):
	if main_config.answers[i][0] == 'OFF':
		offensive_comments.append(comments[i])

uncased_offensive_comments = [z.lower() for z in offensive_comments]

scores = weak_signals.model_aggregator(
            comments=offensive_comments,
            uncased_comments=uncased_offensive_comments,
            task='a')

print([(offensive_comments, _) for _, offensive_comments in sorted(zip(scores, offensive_comments))])
