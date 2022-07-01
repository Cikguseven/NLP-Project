# NLP Project
Using Weak Supervision to perform Named Entity Recognition (NER) and Offensive Language Identification (OLI) on r/Singapore,EDMW and OLID comments.

#### Quick start: 
`pip install requirements.txt` in NER/OLID folder.

## NER

### Workflow to obtain fine-tuned NER model using weak supervision/gold labels:
To scrape comments for NER, create a config.py file in the NER folder with Reddit API info. (Refer to `NER/example_config.py`)
Adjust parameters from `NER/NER_v7/main_config.py`

`NER/NER_v7/make_spacy_weak_supervision.py` (pipeline to scrape and preprocess comments, apply and resolve aggregated labelling functions, serialize .spacy binary file for fine-tuning)  
`NER/NER_cli_train.ipynb` (jupyter notebook to fine-tune model in command line using serialized file)  
`NER/NER_v7/evaluate.py` (test performance on NER task from best model saved to disk after fine-tuning)

### Performance

#### Tested on 500 [r/Singapore](https://www.reddit.com/r/singapore/) comments:
|Fine-tuned model|Precision|Recall|Micro F1|
| -------- | ------- | ---- | ---- |
|Weak Supervision|80.1|81.2|80.6|
|Base spaCy transformer|84.5|78.9|81.6|
|Gold labels|85.3|83.7|84.5|  
  
## OLI  

### Workflow to obtain fine-tuned OLID model using weak supervision/gold labels:  
`OLID/ws_v0/download_transformer_pipeline.py` (download hugging face transformers required for weak supervision pipeline)  
`OLID/ws_v0/make_spacy.py` (pipeline to preprocess comments, apply and resolve aggregated labelling functions, serialize .spacy binary file for fine-tuning)  
`NER/NER_cli_train.ipynb` (jupyter notebook to fine-tune model in command line using serialized file)  
`NER/NER_v7/*_analysis.py` (test OLID performance on chosen dataset from best model saved to disk after fine-tuning) 

### Performance

#### Tested on [OLID dataset](https://sites.google.com/site/offensevalsharedtask/olid):

##### Task A
|Fine-tuned model|F1 OFF (240)|F1 NOT (620)|Macro F1 (840)|
| --- | --- | --- | --- |
|Weak Supervision|72.5|90.0|81.2|
|Weak Supervision (Small)|69.6|87.9|78.7|
|Gold labels|71.9|89.1|80.5|
|Gold labels (Small)|65.2|86.1|75.6| 
|CNN|70.0|90.0|80.0|
 

##### Task B
|Fine-tuned model|F1 TIN (213)|F1 UNT (27)|Macro F1 (240)|
| --- | --- | --- | --- |
|Weak Supervision|91.4|28.6|60.0|
|Weak Supervision (Small)|91.4|28.6|60.0|
|Gold labels|80.0|30.2|55.1|
|Gold labels (Small)|94.0|24.2|59.1| 
|CNN|92.0|42.0|67.0|


##### Task C
|Fine-tuned model|F1 IND (100)|F1 GRP (78)|F1 OTH (35)|Macro F1 (213)|
| --- | --- | --- | --- | --- |
|Weak Supervision|77.3|61.9|25.9|55.0|
|Weak Supervision (Small)|74.7|56.2|17.3|49.4|
|Gold labels|72.6|46.7|27.7|49.0|
|Gold labels (Small)|71.4|43.4|17.8|44.2|
|CNN|75.0|67.0|0.0|47.0|


#### Tested on 1000 [EDMW](https://forums.hardwarezone.com.sg/forums/eat-drink-man-woman.16/) comments:

##### Task A
|Fine-tuned model|F1 OFF (116)|F1 NOT (84)|Macro F1 (200)|
| --- | --- | --- | --- |
|Weak Supervision labels|87.3|81.3|84.3|
|Gold labels|87.7|84.4|86.1|

##### Task B
|Fine-tuned model|F1 TIN (91)|F1 UNT (25)|Macro F1 (116)|
| --- | --- | --- | --- |
|Weak Supervision labels|86.8|46.5|66.6|
|Gold labels|88.5|60.0|74.3|

##### Task C
|Fine-tuned model|F1 IND (36)|F1 GRP (44)|F1 OTH (11)|Macro F1 (91)|
| --- | --- | --- | --- | --- |
|Weak Supervision labels|62.7|39.3|16.7|39.6|
|Gold labels|68.4|79.1|0.0|49.2|


#### Tested on 1000 [r/Singapore](https://www.reddit.com/r/singapore/) comments:

##### Task A
|Fine-tuned model|F1 OFF (62)|F1 NOT (138)|Macro F1 (200)|
| --- | --- | --- | --- |
|Weak Supervision|65.5|87.2|76.3|
|Gold labels|63.7|86.4|75.1|

##### Task B
|Fine-tuned model|F1 TIN (52)|F1 UNT (10)|Macro F1 (62)|
| --- | --- | --- | --- |
|Weak Supervision labels|91.1|66.7|78.9|
|Gold labels|86.0|48.0|67.0|

##### Task C
|Fine-tuned model|F1 IND (18)|F1 GRP (19)|F1 OTH (15)|Macro F1 (52)|
| --- | --- | --- | --- | --- |
|Weak Supervision labels|64.0|46.7|25.0|45.2|
|Gold labels|73.7|48.5|42.4|54.9|  

## Web Application

### Workflow:
Open command prompt in `/flask` and key in the command `npm run dev`
Run `flask_axios_api.py`
Open `127.0.0.1:1234` or `localhost:1234` in a web browser to access the web application
