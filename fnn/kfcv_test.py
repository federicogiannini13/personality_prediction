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

specific_models = config.specific_models or config.embedding_name.endswith("specific")

if specific_models:
    base_root = os.path.join(base_root, "specific_models")
else:
    base_root = os.path.join(base_root, "unique_model")

dl = data_loader.Data_Loader(
    traits=config.ocean_traits,
    embedding_name=config.embedding_name,
    k_folds=config.folds_number,
    words_to_select=words_to_select,
)

models = []
if specific_models:
    for _ in range(0, len(config.ocean_traits)):
        models.append([fnn_model.FNNModel() for _ in range(0, config.folds_number)])
else:
    models = [fnn_model.FNNModel(n_traits=len(config.ocean_traits)) for _ in range(0, config.folds_number)]

file_suffix = (
    "_" + str(config.folds_number) + "fcv_" + config.embedding_name + embedding_dict_str
)

for fold in range(0, config.folds_number):
    if specific_models:
        for cont_tr, trait in enumerate(config.ocean_traits):
            dl.data[cont_tr].initialize_data_kfolds_cv(fold)
            root_ = os.path.join(base_root, str(trait) + "_trait", str(fold) + "_fold")

            m = models[cont_tr][fold]

            m.fit_predict(
                dl.data[cont_tr].train_inputs,
                dl.data[cont_tr].train_outputs[trait],
                dl.data[cont_tr].test_inputs,
                dl.data[cont_tr].test_outputs[trait],
                epochs=config.epochs,
                batch_size=config.batch_size,
                root=root_,
            )
            best_epoch = np.argmax(m.r2)
            df_performance = df_performance.append(
                {
                    "trait": trait,
                    "fold": fold,
                    "best_epoch": best_epoch+1,
                    "r2": m.r2[best_epoch],
                    "mse": m.mse[best_epoch],
                },
                ignore_index=True,
            )
    else:
        dl.data[0].initialize_data_kfolds_cv(fold)
        train_outputs = np.asarray(dl.data[0].train_outputs).take(config.ocean_traits, axis=0).transpose()
        test_outputs = np.asarray(dl.data[0].test_outputs).take(config.ocean_traits, axis=0).transpose()
        root_ = os.path.join(base_root, str(fold) + "_fold")
        m = models[fold]
        m.fit_predict(
            dl.data[0].train_inputs,
            train_outputs,
            dl.data[0].test_inputs,
            test_outputs,
            epochs=config.epochs,
            batch_size=config.batch_size,
            root=root_,
        )

        for cont_tr, trait in enumerate(config.ocean_traits):
            df_performance = df_performance.append(
                {
                    "trait": trait,
                    "fold": fold,
                    "best_epoch": m.best_epoch+1,
                    "r2": m.r2[cont_tr][m.best_epoch],
                    "mse": m.mse[cont_tr][m.best_epoch],
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
