from detoxify import Detoxify
from hatesonar import Sonar
# import main_config
# import comment_filter

detoxify_model = Detoxify('unbiased', device='cuda')

sonar_model = Sonar()

# filtered_cased_comments = comment_filter.c_filter(
#     shuffle=False,
#     remove_username=False,
#     remove_commas=True,
#     length_min=0,
#     length_max=99,
#     uncased=False,
#     unique=False,
#     input_file=main_config.hand_labelled_comments)

sentence = "What the fuck."

sentence_1 = 'he is a gay faggot'

print(detoxify_model.predict(sentence))

print(detoxify_model.predict(sentence_1))

print(sonar_model.ping(text=sentence)['classes'])
print(sonar_model.ping(text=sentence_1)['classes'])

# detoxify insult value
# hatesonar hate vs ol