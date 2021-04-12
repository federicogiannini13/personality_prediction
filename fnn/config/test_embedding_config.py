# ______
# IMPORT:
import os
import sys
from utils import load_yaml_config, Config

sys.path.insert(0, "../../")

# ______
# CONFIG:
config = Config()
config.k = 5
# number of nearest neighbor of KNN algorithm.
config.embedding_name = "glove"
# the embedding to be used. There must be a directory containing the embedding in data folder.
config.ocean_traits = [0, 1, 2, 3, 4]
# OCEAN personality traits to which perform the coherence test: O:0, C:1, E:2, A:3, N:4
config.OUTPUTS_DIR = None
# The base path in which tests' outputs will be saved. Set as None if you want to store them in project's dir.

config = load_yaml_config(
    config,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_embdding_config.yaml"
    ),
)
