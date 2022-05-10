olid_directory = '../OLID_dataset/'
model_directory = '../models/'
NER_model = '../../NER/models/v7/model-best'

training_tweet_file = olid_directory + 'olid-training-v1.tsv'
balanced_tweet_file = olid_directory + 'olid-combined.tsv'

test_tweets_a_file = olid_directory + 'testset-levela.tsv'
test_tweets_b_file = olid_directory + 'testset-levelb.tsv'
test_tweets_c_file = olid_directory + 'testset-levelc.tsv'
test_tweet_files = (test_tweets_a_file, test_tweets_b_file, test_tweets_c_file)

test_answers_a_file = olid_directory + 'labels-levela.csv'
test_answers_b_file = olid_directory + 'labels-levelb.csv'
test_answers_c_file = olid_directory + 'labels-levelc.csv'
test_answer_files = (test_answers_a_file,
                     test_answers_b_file, test_answers_c_file)

handlabelled_reddit_comments = 'redditsg_testing.txt'
handlabelled_hwz_comments = 'hwz_testing.txt'

validation_split = 0.25

version = 'wk13_ws_hwz_45a_12b'

spacy_training_file = version + '_training.spacy'
spacy_validation_file = version + '_vaildation.spacy'


def training_tweets_getter(unlabelled: bool):
    with open(training_tweet_file, encoding='utf-8') as f:
        tweets = [line.split('\t') for line in f]

    tweets.pop(0)
    for tweet in tweets:
        tweet[4] = tweet[4].strip()

    if unlabelled:        
        output_tweets = [tweet[1] for tweet in tweets]
        return output_tweets
    else:
        return tweets


def balanced_tweets_getter(analysis_set: bool):
    with open(balanced_tweet_file, encoding='utf-8') as f:
        tweets = [line.split('\t') for line in f]

    tweets.pop(0)

    if analysis_set:
        for tweet in tweets:
            tweet[4] = tweet[4].strip()

        nn_tweets = []
        nn_counter = 0

        ou_tweets = []

        ot_tweets = []
        ot_counter = 0

        ind_tweets = []
        ind_counter = 0

        grp_tweets = []
        grp_counter = 0

        oth_tweets = []

        for tweet in tweets:
            if nn_counter < 1102 and tweet[2] == 'NOT':
                nn_tweets.append(tweet[1])
                nn_counter += 1
            elif tweet[3] == 'UNT':
                ou_tweets.append(tweet[1])
            elif ot_counter < 551 and tweet[3] == 'TIN':
                ot_tweets.append(tweet[1])
                ot_counter += 1

            if tweet[4] == 'OTH':
                oth_tweets.append(tweet[1])
            elif tweet[4] == 'IND' and ind_counter < 430:
                ind_tweets.append(tweet[1])
                ind_counter += 1
            elif tweet[4] == 'GRP' and grp_counter < 430:
                grp_tweets.append(tweet[1])
                grp_counter += 1

        output_tweets = nn_tweets + ou_tweets + \
            ot_tweets + oth_tweets + ind_tweets + grp_tweets

        return output_tweets

    else:
        output_tweets = [tweet[1] for tweet in tweets]
        return output_tweets


def test_tweets_getter():
    tweets = []

    for file in test_tweet_files:
        with open(file, encoding='utf-8') as f:
            tweets += [line.split('\t')[1]
                       for line in f if line.split('\t')[1] != 'tweet\n']

    return tweets


def answers_getter():
    answers = []

    for file in test_answer_files:
        with open(file) as f:
            answers += [line.split(',')[1].strip() for line in f]

    return answers


def labelled_comments_getter(file: str, train_test: str):

    if train_test == 'train':
        start = 0
        end = 800
    else:
        start = 800
        end = 1000

    with open(file, encoding='utf-8') as f:
        comments = [line.split('|') for line in f][start:end]

    comments = [[x.strip() for x in comment] for comment in comments]

    nn_comments = []
    ou_comments = []
    ind_comments = []
    grp_comments = []
    oth_comments = []

    nn_counter = 0
    ou_counter = 0
    ind_counter = 0
    grp_counter = 0
    oth_counter = 0

    for comment in comments:
        if comment[1] == 'NOT':
            nn_comments.append(comment[0])
            nn_counter += 1
        elif comment[2] == 'UNT':
            ou_comments.append(comment[0])
            ou_counter += 1
        elif comment[3] == 'IND':
            ind_comments.append(comment[0])
            ind_counter += 1
        elif comment[3] == 'GRP':
            grp_comments.append(comment[0])
            grp_counter += 1
        elif comment[3] == 'OTH':
            oth_comments.append(comment[0])
            oth_counter += 1

    output_comments =  nn_comments + ou_comments + \
        ind_comments + grp_comments + oth_comments

    output_distribution = [nn_counter, ou_counter, ind_counter, grp_counter, oth_counter]

    return output_comments, output_distribution
