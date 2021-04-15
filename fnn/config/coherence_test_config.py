# ______
# IMPORT:
from utils import load_yaml_config, Config
import os

while not os.getcwd().endswith("personality_prediction"):
    os.chdir(os.path.dirname(os.getcwd()))

# ______
# CONFIG:
config = Config()
config.ocean_traits = [0, 1, 2, 3, 4]
# OCEAN personality traits to which perform the coherence test: O:0, C:1, E:2, A:3, N:4.
config.distances = [0, 4]
# Distances to which perform the coherence test.
config.max_neigs = 12500
# Maximum number of unknown neighbors to return in the case of distance>0.
# Use None or 0 if you want to return all possible neighbors in the select distance.
config.batch_size = 32
# Training batch size of fnn models.
config.epochs = [50, 300]
config.epochs_train2 = 300
config.epochs_interval = 50
# Epochs is a list of len=2 containing the range of epochs after which stop training of M1 models and train a new model M2.
# M1's training will stop after epochs[0]+n*interval such that  n>0 and epochs[0]+n*interval<=epochs[1]
# M2's training will last epochs_train2 epochs.
config.epochs_interval_evaluation = 1
# M2's training will stop epochs_interval_evaluation epochs to evaluate performance
# M1's training will stop to evaluate performance only if test1=True
config.folds_number = 10
# Numbers of K-fold CV folds.
config.embedding_name = "tuned_embedding"
# The embedding to be used. There must be a directory containing the embedding in data folder.
config.test1 = False
# True if you want to evaluate M1's performances trainings on test set. Use False to skip the evaluation.
config.OUTPUTS_DIR = None
# The base path in which tests' outputs will be saved. Set as None if you want to store them in project's dir.
config.embedding_dict_to_use = None
# If you want to use the dictionary of another embedding, set this parameter with the embedding name. Use None otherwise.
# There must be a directory containing the embedding in data folder.

config = load_yaml_config(
    config,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "coherence_test_config.yaml"
    ),
)
