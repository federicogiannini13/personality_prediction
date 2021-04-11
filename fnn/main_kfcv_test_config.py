# ______
# IMPORT:
from settings import ROOT_DIR
import sys

sys.path.insert(0, "../")

# ______
# CONFIG:
ocean_traits = [0, 1, 2, 3, 4]
# OCEAN personality traits to which perform the coherence test: O:0, C:1, E:2, A:3, N:4
batch_size = 32
# training batch size of fnn models.
folds_number = 10
# numbers of K-fold CV folds.
embedding_name = "glove"
# the embedding to be used. There must be a directory containing the embedding in data folder.
epochs = 300
# training's epochs number.
OUTPUTS_DIR = ROOT_DIR
# The base path in which tests' outputs will be saved. Set as ROOT_DIR if you want to store them in project's dir.
