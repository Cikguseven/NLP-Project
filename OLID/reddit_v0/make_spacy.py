from spacy.tokens import DocBin
import comment_filter
import main_config
import spacy
import weak_signals


def spacy_file_creator(
        comments: list,
        gold_labels: list,
        length: int,
        weak_supervision_mode: bool,
        training_validation_split: int,
        output_training: str,
        output_vaildation: str):

    spacy.require_gpu()
    nlp = spacy.load('en_core_web_trf')

    db_training = DocBin()
    db_validation = DocBin()

    if weak_supervision_mode:
        task_a_labels, task_b_labels, task_c_labels = weak_signals.model_aggregator(comments)

        threshold = 

        for i in range(length):
            categories = {
                'offensive': 0.0,
                'targeted': 0.0,
                'individual': 0.0,
                'group': 0.0,
                'other': 0.0
            }

            if task_a_labels[i] > threshold:
                categories['offensive'] = 1.0

                if task_b_labels[i] > threshold:
                    categories['targeted'] = 1.0

                    if task_c_labels[i] == 'IND':
                        categories['individual'] = 1.0

                    elif task_c_labels[i] == 'GRP':
                        categories['group'] = 1.0

                    else:
                        categories['other'] = 1.0

            doc = nlp.make_doc(comments[i])
            doc.cats = categories

            if i < training_validation_split:
                db_validation.add(doc)

            else:
                db_training.add(doc)

    else:
        for i in range(length):
            categories = {
                'offensive': 0.0,
                'targeted': 0.0,
                'individual': 0.0,
                'group': 0.0,
                'other': 0.0
            }

            if gold_labels[i][0] == 'OFF':
                categories['offensive'] = 1.0

                if gold_labels[i][1] == 'TIN':
                    categories['targeted'] = 1.0

                    if gold_labels[i][2] == 'IND':
                        categories['individual'] = 1.0

                    elif gold_labels[i][2] == 'GRP':
                        categories['group'] = 1.0

                    else:
                        categories['other'] = 1.0

            doc = nlp.make_doc(comments[i])
            doc.cats = categories

            if i < training_validation_split:
                db_validation.add(doc)

            else:
                db_training.add(doc)

    db_validation.to_disk(output_vaildation)
    db_training.to_disk(output_training)


if __name__ == '__main__':
    import time
    start = time.time()

    olid_training_data = main_config.training_tweets_getter(unlabelled=True)

    # Import unique filtered comments for testing and validation
    filtered_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=olid_training_data)[:]

    comment_count = len(filtered_comments)

    print(comment_count)

    spacy_file_creator(
        comments=filtered_comments[:],
        gold_labels=main_config.answers[:900],
        length=comment_count,
        weak_supervision_mode=True,
        training_validation_split=int(main_config.validation_split * comment_count),
        output_training=main_config.spacy_training_file,
        output_vaildation=main_config.spacy_validation_file)

    end = time.time()
    t = round(end - start, 2)

    if t > 60:
        duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
    else:
        duration = str(t) + 's'

    print(f'Time taken: {duration}')
