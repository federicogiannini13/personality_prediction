import os
import sys
while not os.getcwd().endswith("personality_prediction") and os.getcwd()!="/":
    os.chdir(os.path.dirname(os.getcwd()))
if os.getcwd()=="/":
    raise Exception("The project dir's name must be 'personality_prediction'. Rename it.")
sys.path.append(os.getcwd())

from fnn.config.kfcv_test_config import config
from fnn.modules import data_loader, fnn_model
import pandas as pd
import numpy as np
from settings import ROOT_DIR

df_performance = pd.DataFrame(columns=["trait", "fold", "best_epoch", "mse", "r2"])
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
base_root = os.path.join(
    base_root,
    "KFCV",
    str(config.folds_number) + "_folds",
)

dl = data_loader.Data_Loader(
    traits=config.ocean_traits,
    embedding_name=config.embedding_name,
    k_folds=config.folds_number,
    words_to_select=words_to_select,
)

models = []
for i in range(0, len(config.ocean_traits)):
    models.append([fnn_model.FNNModel() for i in range(0, config.folds_number)])

file_suffix = (
    "_" + str(config.folds_number) + "fcv_" + config.embedding_name + embedding_dict_str
)

for fold in range(0, config.folds_number):
    for cont_tr, trait in enumerate(config.ocean_traits):
        dl.data[cont_tr].initialize_data_kfolds_cv(fold)
        root_ = os.path.join(base_root, str(trait) + "_trait", str(fold) + "_fold")

        m = models[cont_tr][fold]

        m.fit_predict(
            dl.data[cont_tr].train_inputs,
            dl.data[cont_tr].train_outputs[cont_tr],
            dl.data[cont_tr].test_inputs,
            dl.data[cont_tr].test_outputs[cont_tr],
            epochs=config.epochs,
            batch_size=config.batch_size,
            root=root_,
        )
        best_epoch = np.argmax(m.r2)
        r2 = m.r2[best_epoch]
        mse = m.mse[best_epoch]

        df_performance = df_performance.append(
            {
                "trait": trait,
                "fold": fold,
                "best_epoch": best_epoch,
                "r2": r2,
                "mse": mse,
            },
            ignore_index=True,
        )
        df_performance.to_excel(
            os.path.join(
                base_root,
                "performances" + file_suffix + ".xlsx",
            ),
            index=False,
        )

df_performance.groupby("trait").mean().reset_index().drop(columns="fold").to_excel(
    os.path.join(
        base_root,
        "final_performances" + file_suffix + ".xlsx",
    ),
    index=False,
)
