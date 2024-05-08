## FRELSA dataframe

This repository makes reference to the article 'XXXXXX' and it is intended to create the FRELSA database described in sais article, and train some simple architectures for frailty detection and prediction scopes.

To use the repository one must have access to the ELSA data (see portal https://www.elsa-project.ac.uk/)

Run main.py to create the frailty label.
Run preprocess.py to select best features for detection and prediction (WARNING this operation is costly and requires parallel jobs).
Run models.py to train models and save the results.
