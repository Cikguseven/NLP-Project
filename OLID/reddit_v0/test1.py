from bad_words import offensive_lexicon

z = 'fuckll'

if any(offensive_word in z for offensive_word in offensive_lexicon):
	print('a')