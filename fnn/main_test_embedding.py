import os
from fnn import data_loader
import pandas as pd
from fnn.config.main_test_embedding_config import config
from utils import create_dir
import sys

sys.path.insert(0, "../")

df_performance = pd.DataFrame(columns=["trait", "r2"])
base_root = os.path.join(config.OUTPUTS_DIR, "outputs", config.embedding_name, "KNN")
create_dir(base_root)

dl = data_loader.Data_Loader(
    traits=config.ocean_traits,
    train_prop_holdout=1,
    embedding_name=config.embedding_name,
    standardize_holdout=False,
    shuffle=False,
)

for cont_tr, trait in enumerate(config.ocean_traits):
    print(trait, " TRAIT")
    r2 = dl.data[cont_tr].embedding_test(trait=trait, k=config.k)
    print(r2)
    df_performance = df_performance.append(
        {"trait": trait, "r2": r2}, ignore_index=True
    )
    df_performance.to_excel(
        os.path.join(base_root, str(config.k) + "nn_performances.xlsx"), index=False
    )
