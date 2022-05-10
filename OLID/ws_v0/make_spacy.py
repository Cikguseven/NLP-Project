from spacy.tokens import DocBin
import comment_filter
import main_config
import spacy
import weak_signals


def spacy_file_creator(
        comments: list,
        distribution: list,
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

        threshold_a = 0.4
        threshold_b = 0.08

        for i in range(length):
            categories = {
                'offensive': 0.0,
                'targeted': 0.0,
                'individual': 0.0,
                'group': 0.0,
                'other': 0.0
            }

            if task_a_labels[i] > threshold_a:
                categories['offensive'] = 1.0

            if task_b_labels[i] > threshold_b:
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

            if i >= distribution[0]:
                categories['offensive'] = 1.0

                if i >= sum(distribution[:2]):
                    categories['targeted'] = 1.0

                    if i >= sum(distribution[:4]):
                        categories['other'] = 1.0

                    elif i >= sum(distribution[:3]):
                        categories['group'] = 1.0

                    else:
                        categories['individual'] = 1.0

            doc = nlp.make_doc(comments[i])
            doc.cats = categories

            if i < training_validation_split:
                db_validation.add(doc)

            else:
                db_training.add(doc)

    db_validation.to_disk(output_vaildation)
    db_training.to_disk(output_training)


if __name__ == '__main__':
    # # OLID tweets
    # training_comments = main_config.training_tweets_getter(unlabelled=True)

    # Reddit/HWZ comments
    training_comments, distribution = main_config.labelled_comments_getter(
        file=main_config.handlabelled_hwz_comments, train_test='train')

    print(distribution)

    # Import unique filtered comments for testing and validation
    filtered_comments = comment_filter.c_filter(
        shuffle=False,
        remove_username=False,
        remove_commas=False,
        length_min=0,
        length_max=9999,
        uncased=False,
        unique=False,
        input_list=training_comments)

    comment_count = len(filtered_comments)

    is_ws = True

    if is_ws:
        print('Creating weak supervision spacy file')
    else:
        print('Creating finetune spacy file')

    spacy_file_creator(
        comments=filtered_comments[:],
        distribution=distribution,
        length=comment_count,
        weak_supervision_mode=is_ws,
        training_validation_split=int(
            main_config.validation_split * comment_count),
        output_training=main_config.spacy_training_file,
        output_vaildation=main_config.spacy_validation_file)
