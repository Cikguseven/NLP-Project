import main_config
from os import listdir

custom_models = [f for f in listdir(
    main_config.NER_model_directory)]

print(custom_models)

