import os
import sys
while not os.getcwd().endswith("personality_prediction") and os.getcwd()!="/":
    os.chdir(os.path.dirname(os.getcwd()))
if os.getcwd()=="/":
    raise Exception("The project dir's name must be 'personality_prediction'. Rename it.")
sys.path.append(os.getcwd())

# ______
# IMPORT:
from utils import load_yaml_config, Config

# ______
# CONFIG:
config = Config()
config.k = 5
# number of nearest neighbor of KNN algorithm.
config.embedding_name = "tuned_embedding"
# the embedding to be used. There must be a directory containing the embedding in data folder.
config.ocean_traits = [0, 1, 2, 3, 4]
# OCEAN personality traits to which perform the coherence test: O:0, C:1, E:2, A:3, N:4
config.OUTPUTS_DIR = None
# The base path in which tests' outputs will be saved. Set as None if you want to store them in project's dir.
config.embedding_dict_to_use = None
# If you want to use the dictionary of another embedding, set this parameter with the embedding name. Use None otherwise.
# There must be a directory containing the embedding in data folder.

config = load_yaml_config(
    config,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_embdding_config.yaml"
    ),
)
