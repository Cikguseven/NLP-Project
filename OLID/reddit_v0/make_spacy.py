from spacy.tokens import DocBin
import comment_filter
import main_config
import spacy
import time
import weak_signals

start = time.time()

spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')


def spacy_file_creator(
        comments: list,
        gold_labels: list,
        weak_supervision_mode: bool,
        training_validation_split: int,
        output_training: str,
        output_vaildation: str):

    db_training = DocBin()
    db_validation = DocBin()

    comment_count = len(comments)

    uncased_comments = [comment.lower() for comment in comments]

    if weak_supervision_mode:
        weak_labelled_comments = weak_signals.model_aggregator(comments, uncased_comments)

    else:
        for i in range(comment_count):
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

    # Import unique filtered comments for testing and validation
    filtered_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=99,
        uncased=True,
        unique=False,
        input_file=main_config.hand_labelled_comments)

    spacy_file_creator(
        comments=filtered_comments[:900],
        gold_labels=main_config.answers[:900],
        weak_supervision_mode=False,
        training_validation_split=main_config.validation_split,
        output_training=main_config.spacy_training_file,
        output_vaildation=main_config.spacy_validation_file)

    end = time.time()
    t = round(end - start, 2)

    if t > 60:
        duration = str(int(t // 60)) + 'm ' + str(int(t % 60)) + 's'
    else:
        duration = str(t) + 's'

    print(f'Time taken: {duration}')
