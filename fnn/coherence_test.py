import os
import sys
while not os.getcwd().endswith("personality_prediction") and os.getcwd()!="/":
    os.chdir(os.path.dirname(os.getcwd()))
if os.getcwd()=="/":
    raise Exception("The project dir's name must be 'personality_prediction'. Rename it.")
sys.path.append(os.getcwd())

from fnn.config.coherence_test_config import config
from fnn.modules import data_loader, coherence_checker
import pandas as pd
from settings import ROOT_DIR

config.epochs.sort()
df_performance_coherence = pd.DataFrame(
    columns=[
        "trait",
        "fold",
        "epochs_train1",
        "best_epoch_train2",
        "best_r2",
        "best_mse",
    ]
)
if config.test1:
    df_performance_kfolds = pd.DataFrame(
        columns=["trait", "fold", "best_epoch", "mse", "r2"]
    )

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
base_root = os.path.join(base_root, "coherence_test")

for distance in config.distances:
    root = os.path.join(
        base_root, str(distance) + "_dist", str(config.folds_number) + "_folds"
    )
    file_suffix = (
        "_"
        + str(distance)
        + "dist_"
        + str(config.folds_number)
        + "folds_"
        + config.embedding_name
        + embedding_dict_str
    )
    dl = data_loader.Data_Loader(
        traits=config.ocean_traits,
        distance=distance,
        embedding_name=config.embedding_name,
        k_folds=config.folds_number,
        max_neigs=config.max_neigs,
        words_to_select=words_to_select,
    )

    for fold in range(0, config.folds_number):
        for cont_tr, trait in enumerate(config.ocean_traits):
            dl.data[cont_tr].initialize_data_kfolds_cv(fold)
            dl.data[cont_tr].search_unknown_neighbors(
                distance=distance, max_neigs=config.max_neigs
            )
            root_ = os.path.join(
                root,
                str(trait) + "_trait",
                str(fold) + "_fold",
                str(config.epochs[0]) + "_ep",
            )
            checker = coherence_checker.CoherenceChecker(
                inputs=dl.data[cont_tr].train_inputs,
                outputs=[dl.data[cont_tr].train_outputs[trait]],
                inputs_neig=dl.data[cont_tr].inputs_neig,
                test_inputs=dl.data[cont_tr].test_inputs,
                test_outputs=[dl.data[cont_tr].test_outputs[trait]],
                batch_size=config.batch_size,
                ocean_traits=[trait],
                test1=config.test1,
            )

            checker.train1_inference(
                epochs=config.epochs[0],
                root=root_,
                epochs_interval_evaluation=config.epochs_interval_evaluation,
            )
            if config.test1:
                df_performance_kfolds = df_performance_kfolds.append(
                    {
                        "trait": trait,
                        "fold": fold,
                        "best_epoch": checker.best_epochs_train1[0],
                        "r2": checker.best_r2_train1[0],
                        "mse": checker.best_mse_train1[0],
                    },
                    ignore_index=True,
                )
                df_performance_kfolds.to_excel(
                    os.path.join(
                        base_root,
                        "performances_kfolds" + file_suffix + ".xlsx",
                    ),
                    index=False,
                )
            checker.train2_coherence(
                epochs=config.epochs_train2,
                root=root_,
                epochs_interval_evaluation=config.epochs_interval_evaluation,
            )
            df_performance_coherence = df_performance_coherence.append(
                {
                    "trait": trait,
                    "fold": fold,
                    "epochs_train1": config.epochs[0],
                    "best_epoch_train2": checker.best_epochs_train2[0],
                    "best_r2": checker.best_r2_train2[0],
                    "best_mse": checker.best_mse_train2[0],
                },
                ignore_index=True,
            )
            df_performance_coherence.to_excel(
                os.path.join(
                    root,
                    "performances_coherence" + file_suffix + ".xlsx",
                ),
                index=False,
            )

            e = config.epochs[0]
            e += config.epochs_interval
            while e <= config.epochs[1]:
                root_ = os.path.join(
                    root, str(trait) + "_trait", str(fold) + "_fold", str(e) + "_ep"
                )
                checker.train1_inference(
                    reset_models=False,
                    epochs=config.epochs_interval,
                    root=root_,
                    epochs_interval_evaluation=config.epochs_interval_evaluation,
                )
                checker.train2_coherence(
                    epochs=config.epochs_train2,
                    root=root_,
                    epochs_interval_evaluation=config.epochs_interval_evaluation,
                )
                df_performance_coherence = df_performance_coherence.append(
                    {
                        "trait": trait,
                        "fold": fold,
                        "epochs_train1": e,
                        "best_epoch_train2": checker.best_epochs_train2[0],
                        "best_r2": checker.best_r2_train2[0],
                        "best_mse": checker.best_mse_train2[0],
                    },
                    ignore_index=True,
                )
                df_performance_coherence.to_excel(
                    os.path.join(
                        root,
                        "performances_coherence" + file_suffix + ".xlsx",
                    ),
                    index=False,
                )
                e += config.epochs_interval

    df_performance_coherence.groupby(["trait", "fold"]).max(
        "best_r2"
    ).reset_index().groupby("trait").mean().reset_index().drop(columns="fold").to_excel(
        os.path.join(
            root,
            "final_performances_coherence" + file_suffix + ".xlsx",
        ),
        index=False,
    )

    if config.test1:
        df_performance_kfolds.groupby("trait").mean().reset_index().drop(
            columns="fold"
        ).to_excel(
            os.path.join(
                root,
                "final_performances_coherence" + file_suffix + ".xlsx",
            ),
            index=False,
        )
