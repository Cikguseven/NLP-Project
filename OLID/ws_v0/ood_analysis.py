import comment_filter
import main_config
import spacy

with open('hwz.csv', encoding='utf-8') as f:
	hwz_comments = [line.split(',', 2)[2].strip() for line in f]

hwz_comments.pop(0)

# Import unique filtered comments for testing
filtered_hwz_comments = comment_filter.c_filter(
    shuffle=False,
    remove_username=False,
    remove_commas=False,
    length_min=10,
    length_max=9999,
    uncased=False,
    unique=True,
    input_list=hwz_comments)

# Import unique filtered comments for testing
filtered_gab_comments = comment_filter.c_filter(
    shuffle=False,
    remove_username=False,
    remove_commas=False,
    length_min=10,
    length_max=9999,
    uncased=False,
    unique=True,
    input_file='gab_comments.txt')
    
models = [f for f in listdir(main_config.model_directory) if 'b_' in f]

for model in models:

    print(model)

    nlp = spacy.load(main_config.model_directory +
                        model + '/model-best')

    docs = list(nlp.pipe(filtered_hwz_comments))

    task_a_predictions = [doc.cats['offensive'] for doc in docs]
    task_b_predictions = [doc.cats['targeted'] for doc in docs]
    task_c_ind_predictions = [doc.cats['individual'] for doc in docs]
    task_c_grp_predictions = [doc.cats['group'] for doc in docs]
    task_c_oth_predictions = [doc.cats['other'] for doc in docs]

    