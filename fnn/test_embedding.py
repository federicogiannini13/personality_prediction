import os
import sys
while not os.getcwd().endswith("personality_prediction") and os.getcwd()!="/":
    os.chdir(os.path.dirname(os.getcwd()))
if os.getcwd()=="/":
    raise Exception("The project dir's name must be 'personality_prediction'. Rename it.")
sys.path.append(os.getcwd())

from fnn.modules import data_loader
import pandas as pd
from fnn.config.test_embedding_config import config
from settings import ROOT_DIR
from utils import create_dir

df_performance = pd.DataFrame(columns=["trait", "r2"])
base_root = os.path.join(config.OUTPUTS_DIR, "outputs", config.embedding_name)

if config.embedding_dict_to_use is not None:
    import pickle

    embedding_path = os.path.join(ROOT_DIR, "data", config.embedding_dict_to_use)
    with open(os.path.join(embedding_path, "words.pickle"), "rb") as f:
        words_to_select = pickle.load(f)
    embedding_dict_str = config.embedding_dict_to_use + "_dict"
    base_root = os.path.join(base_root, embedding_dict_str)
    embedding_dict_str = "_" + embedding_dict_str
else:
    words_to_select = None
    embedding_dict_str = ""
base_root = os.path.join(base_root, "KNN")
create_dir(base_root)

dl = data_loader.Data_Loader(
    traits=config.ocean_traits,
    train_prop_holdout=1,
    embedding_name=config.embedding_name,
    standardize_holdout=False,
    shuffle=False,
    words_to_select=words_to_select,
)

for cont_tr, trait in enumerate(config.ocean_traits):
    print(trait, " TRAIT")
    r2 = dl.data[cont_tr].embedding_test(trait=trait, k=config.k)
    print(r2)
    df_performance = df_performance.append(
        {"trait": trait, "r2": r2}, ignore_index=True
    )
    df_performance.to_excel(
        os.path.join(
            base_root,
            "performances_"
            + str(config.k)
            + "nn_"
            + config.embedding_name
            + embedding_dict_str
            + ".xlsx",
        ),
        index=False,
    )
