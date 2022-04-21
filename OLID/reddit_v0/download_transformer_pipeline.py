from transformers import pipeline

tc_0 = pipeline(task='text-classification', model="siebert/sentiment-roberta-large-english")
tc_0.save_pretrained('./pipelines/tc_0')

tc_1 = pipeline(task='text-classification', model="distilbert-base-uncased-finetuned-sst-2-english")
tc_1.save_pretrained('./pipelines/tc_1')

tc_2 = pipeline(task='text-classification', model="mrm8488/distilroberta-finetuned-tweets-hate-speech")
tc_2.save_pretrained('./pipelines/tc_2')

tc_3 = pipeline(task='text-classification', model="cardiffnlp/twitter-roberta-base-offensive")
tc_3.save_pretrained('./pipelines/tc_3')

tc_4 = pipeline(task='text-classification', model="elozano/tweet_offensive_eval")
tc_4.save_pretrained('./pipelines/tc_4')

tc_5 = pipeline(task='text-classification', model="IMSyPP/hate_speech_en")
tc_5.save_pretrained('./pipelines/tc_5')

tc_6 = pipeline(task='text-classification', model="Narrativaai/deberta-v3-small-finetuned-hate_speech18")
tc_6.save_pretrained('./pipelines/tc_6')

tc_7 = pipeline(task='text-classification', model="Hate-speech-CNERG/dehatebert-mono-english")
tc_7.save_pretrained('./pipelines/tc_7')

tc_8 = pipeline(task='text2text-generation', model="Narrativa/byt5-base-tweet-hate-detection")
tc_8.save_pretrained('./pipelines/tc_8')
