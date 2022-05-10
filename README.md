# NLP-Project
Using weak supervision to perform named entity recognition and offensive language detection on r/sg comments

### Performance benchmark of NER task on 500 reddit comments:
|Model     |Precision|Recall|F1 (Micro)|
| -------- | ------- | ---- | ---- |
|This Model|79.3   |78.1|78.7|
|default spaCy transformer|86.3|76.9|81.3|  
  
#### Workflow to obtain weak supervision/fine tuned spaCy transformer model:  
`NER/NER_v7/make_spacy_weak_supervision.py` (pipeline to scrape and preprocess comments, apply and resolve aggregated labelling functions, serialize file for training)  
`NER/NER_cli_train.ipynb` (jupyter notebook to train model in command line using serialized file)  
`NER/NER_v7/evaluate.py` (test performance on NER task from best model saved to disk after training)  
