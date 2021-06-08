# PersonalityPrediction
This repository contains the code used for the experimentation shown in the paper.

### Installation
1) Download the file from Google Drive and unzip it to the project dir.
2) Download `yelp_academic_dataset_review.json` from https://www.yelp.com/dataset/download and put it in the `data/yelp_dataset` folder.
3) execute `pip install -r requirements.txt` .
4) execute `python -m embedding_tuning.nltk_init` .

### Project structure
The project is composed by the following directories.
#### data
Contains embeddings (GloVe and tuned embeddings), known terms scoring dataset and the Yelp reviews corpus.
#### embedding_tuning
Module that tunes GloVe embedding.
* **tune_embedding.py**: tunes GloVe embedding by training the CNN models on Yelp reviews corpus. Saves the tuned embeddings in the data directory. Each trait i (i=0: trait O, i=1: trait C, i=2: trait E, i=3: trait A, i=4: trait N) has its own `i_trait` subdirecotry containing the pickle file representing the hidden layer weights' matrix associated to the CNN model of the trait i. It stores also other pickle files: embedding vocabulary, words' frequencies, test/train inputs/outputs and initial weights.
#### fnn
Module that trains fnn models and performs performance tests (coherence, kfcv and embedding test).
* **coherence_test.py**: performs the coherence test on the specified embedding. Results are stored in `outputs/embedding/coherence_test/k_folds/d_dist/final_performances_coherence_(d)dist_(k)folds_(embedding).xlsx`, where embedding is the specified embedding, k is the specified number of folds and d is the specified type of neighbors' set.
* **kfcv_test.py**: performs the K-folds Cross-Validation test on the specified embedding. Results are stored in `outputs/(embedding)/KFCV/(k)_folds/final_performances_(k)fcv_(embedding).xlsx`, where embedding is the specified embedding and k is the specified number of folds.
* **predict.ipynb**: notebook that trains the five fnn models on all the known terms and stores unknown terms' marker indices associated to the five traits in `outputs/(embedding)/predictions.xlsx`, where embedding is the specified embedding.
* **test_embedding.py**: performs the KNN test on the specified embedding. Performances are stored in `outputs/(embedding)/KNN/performances_(k)nn_(embedding).xlsx` where embedding is the specified embedding and k is the specified number of nearest neighbors.

### Running configuration
Each module has its own `config` subfolder that contains a file, whose name ends with `_config.py`, for each of the specified py files in the Project structure paragraph.
To specify, for the first time, the running configuration of a specific py file, run the associated `_config.py` file. A yaml file with the same name will be created. Edit the yaml file with the desired running configuration and run the py file. The meaning of each parameter is specified in `_config.py`'s comments.
