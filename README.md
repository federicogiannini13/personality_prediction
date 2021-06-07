# PersonalityPrediction
This repository contains the code used for the experimentation shown in the paper.

### Installation
1) execute `pip install -r requirements.txt` .
2) execute `python -m embedding_tuning.nltk_init` .
3) Download the file from Google Drive and unzip it to the project dir.

### Project structure
The project is composed by the following directories.
#### data
Contains embeddings (GloVe, tuned embedding) and known terms scoring dataset.
#### embedding_tuning
Module that tunes GloVe embedding.
* **tune_embedding.py**: tunes GloVe embedding by training the CNN models on Yelp reviews corpus. Saves the tuned embeddings in the data directory. Each trait i (i=0: trait O, i=1: trait C, i=2: trait E, i=3: trait A, i=4: trait N) has its own i_trait subdirecotry containing the pickle file representing the hidden layer weights' matrix associated to the CNN model of trait i. It stores also othe pickle files: embedding vocabulary, words' frequencies, test/train inputs/outputs, initial weights.
#### fnn
Module that trains fnn models and performs performance tests (coherence, kfcv and embedding test)
* **coherence_test.py**: performs the coherence test on the specified embedding. Results are stored in `outputs/embedding/coherence_test/k_folds/d_dist/final_performances_coherence.xlsx`, where embedding is the specified embedding, k is the specified number of folds and d is the specified type of neighbors' set.
* **kfcv_test.py**: performs the K-folds Cross-Validation test on the specified embedding. Results are stored in `outputs/(embedding)/KFCV/(k)_folds/final_performances.xlsx`, where embedding is the specified embedding and k is the specified number of folds.
* **predict.ipynb**: notebook that trains the five fnns on all the known terms and stores unknown terms with marker indices associated to the five traits in `outputs/(embedding)/predictions.xlsx`, where embedding is the specified embedding.
* **test_embedding.py**: performs the KNN test on the specified embedding. Performances are stored in `outputs/(embedding)/KNN/(k)nn_performances.xlsx` where embedding is the specified embedding and k is the specified number of nearest neighbors.

### Running configuration
Each module has its own `config` subfloder that contains a file, whose name ends with `_config.py`, for each of the specified files in the Project structure paragraph.
To specify, for the first time, the running configuration of a specific py file, run the associated `_config.py` file. A yaml file with the same name will be created. Edit the yaml file with the desired running configuration and run the py file. The meaning of each parameter is specified in `_config.py`'s comments.
