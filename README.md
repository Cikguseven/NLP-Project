# NLP-Project
Using weak supervision to perform named entity recognition and offensive language detection on r/Singapore comments

#### Quick start: 
`pip install requirements.txt` in NER/OLID folder.

To scrape comments for NER, create a config.py file in the NER folder with Reddit API info. (Refer to `NER/example_config.py`)

## NER Performance

### Tested on 500 r/Singapore comments:
|Model     |Precision|Recall|F1 (Micro)|
| -------- | ------- | ---- | ---- |
|Weak Supervision|80.1|81.2|80.6|
|Default spaCy transformer|84.5|78.9|81.6|
|Fine tuned spaCy transformer|85.3|83.7|84.5|  
  
#### Workflow to obtain weak supervision/fine tuned spaCy transformer model:  
`NER/NER_v7/make_spacy_weak_supervision.py` (pipeline to scrape and preprocess comments, apply and resolve aggregated labelling functions, serialize file for training)  
`NER/NER_cli_train.ipynb` (jupyter notebook to train model in command line using serialized file)  
`NER/NER_v7/evaluate.py` (test performance on NER task from best model saved to disk after training)  


### OLID Performance

### Tested on [OLID dataset](https://sites.google.com/site/offensevalsharedtask/olid):

#### Task A
|Model|F1 NOT|F1 OFF|F1 Macro|
| --- | --- | --- | --- |
|Weak Supervision|88.2|72.3|80.3|
|CNN|90.0|70.0|80.0|
|Fine tuned spaCy transformer|89.2|73.3|81.3|  

#### Task B
|Model|F1 UNT|F1 TIN|F1 Macro|
| --- | --- | --- | --- |
|Weak Supervision|40.5|87.8|64.2|
|CNN|42.0|92.0|67.0|
|Fine tuned spaCy transformer|13.8|94.2|54.0|  

#### Task C
|Model|F1 IND|F1 GRP|F1 OTH|F1 Macro|
| --- | --- | --- | --- | --- |
|Weak Supervision|76.2|62.7|5.4|48.1|
|CNN|75.0|67.0|0|47.0|
|Fine tuned spaCy transformer|77.6|64.5|0|47.4|

### Tested on 1000 r/Singapore comments::

#### Task A
|Model|F1 NOT|F1 OFF|F1 Macro|
| --- | --- | --- | --- |
|Weak Supervision|87.3|57.4|72.4|
|Fine tuned spaCy transformer|87.1|59.1|73.2|  

#### Task B
|Model|F1 UNT|F1 TIN|F1 Macro|
| --- | --- | --- | --- |
|Weak Supervision|50.0|97.1|73.5|
|Fine tuned spaCy transformer|50.0|97.1|73.5|  

#### Task C
|Model|F1 IND|F1 GRP|F1 OTH|F1 Macro|
| --- | --- | --- | --- | --- |
|Weak Supervision|60.9|46.7|25.0|44.2|
|Fine tuned spaCy transformer|70.6|48.5|42.2|53.8|  
  
#### Workflow to obtain weak supervision/fine tuned spaCy transformer model:  
`OLID/ws_v0/download_transformer_pipeline.py` (download hugging face transformers required for weak supervision pipeline)  
`OLID/ws_v0/make_spacy.py` (pipeline to preprocess comments, apply and resolve aggregated labelling functions, serialize file for training)  
`NER/NER_cli_train.ipynb` (jupyter notebook to train model in command line using serialized file)  
`NER/NER_v7/*_analysis.py` (test OLID performance on chosen dataset from best model saved to disk after training) 
