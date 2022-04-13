import collections
import emoji

olid_directory = '../OLID_dataset/'
model_directory = '../models/'

training_tweets = olid_directory + 'olid-training-v1.tsv'

test_tweets_a = olid_directory + 'testset-levela.tsv'
test_tweets_b = olid_directory + 'testset-levelb.tsv'
test_tweets_c = olid_directory + 'testset-levelc.tsv'

test_answers_a = olid_directory + 'labels-levela.csv'
test_answers_b = olid_directory + 'labels-levelb.csv'
test_answers_c = olid_directory + 'labels-levelc.csv'

combined_tweets = []
test_tweet_files = (test_tweets_a, test_tweets_b, test_tweets_c)

for file in test_tweet_files:
	with open(file, encoding='utf-8') as f:
		combined_tweets += [emoji.demojize(line.split('\t')[1].strip(), delimiters=(' ', ' ')) for line in f if line.split('\t')[1] != 'tweet\n']


combined_answers = []
test_answer_files = (test_answers_a, test_answers_b, test_answers_c)

for file in test_answer_files:
	with open(file) as f:
		combined_answers += [line.split(',')[1].strip() for line in f]

category_frequency = collections.Counter(combined_answers)

validation_split = 25

version = 'v1_'

output_training = version + 'olid_training.spacy'
output_vaildation = version + 'olid_vaildation.spacy'



