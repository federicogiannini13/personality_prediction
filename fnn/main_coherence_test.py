import os
from fnn.main_coerence_test_config import *
from fnn import data_loader, coherence_checker
import pandas as pd
import sys

sys.path.insert(0, "../")

epochs.sort()
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
if test1:
    df_performance_kfolds = pd.DataFrame(
        columns=["trait", "fold", "best_epoch", "mse", "r2"]
    )
base_root = os.path.join(
    OUTPUTS_DIR,
    "outputs",
    embedding_name,
    "coherence_test",
    str(folds_number) + "_folds",
)

for distance in distances:
    root = os.path.join(base_root, str(distance) + "_dist")
    dl = data_loader.Data_Loader(
        traits=ocean_traits,
        distance=distance,
        embedding_name=embedding_name,
        k_folds=folds_number,
    )
    fold = 0

    for fold in range(0, folds_number):
        for cont_tr, trait in enumerate(ocean_traits):
            dl.data[cont_tr].initialize_data_kfolds_cv(fold)
            dl.data[cont_tr].search_unknown_neighbors(distance=distance)
            root_ = os.path.join(
                root, str(trait) + "_trait", str(fold) + "_fold", str(epochs[0]) + "_ep"
            )
            checker = coherence_checker.CoherenceChecker(
                inputs=dl.data[cont_tr].train_inputs,
                outputs=[dl.data[cont_tr].train_outputs[trait]],
                inputs_neig=dl.data[cont_tr].inputs_neig,
                test_inputs=dl.data[cont_tr].test_inputs,
                test_outputs=[dl.data[cont_tr].test_outputs[trait]],
                batch_size=batch_size,
                ocean_traits=[trait],
                test1=test1,
            )

            checker.train1_inference(epochs=epochs[0], root=root_)
            if test1:
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
                    os.path.join(base_root, "performances_kfolds.xlsx"), index=False
                )
            checker.train2_coherence(epochs=epochs[1], root=root_)
            df_performance_coherence = df_performance_coherence.append(
                {
                    "trait": trait,
                    "fold": fold,
                    "epochs_train1": epochs[0],
                    "best_epoch_train2": checker.best_epochs_train2[0],
                    "best_r2": checker.best_r2_train2[0],
                    "best_mse": checker.best_mse_train2[0],
                },
                ignore_index=True,
            )
            df_performance_coherence.to_excel(
                os.path.join(root, "performances_coherence.xlsx"), index=False
            )

            e = epochs[0]
            e += interval
            while e <= epochs[1]:
                root_ = os.path.join(
                    root, str(trait) + "_trait", str(fold) + "_fold", str(e) + "_ep"
                )
                checker.train1_inference(
                    reset_models=False, epochs=interval, root=root_
                )
                checker.train2_coherence(epochs=epochs[1], root=root_)
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
                    os.path.join(root, "performances_coherence.xlsx"), index=False
                )
                e += interval

    df_performance_coherence.groupby(["trait", "fold"]).max(
        "best_r2"
    ).reset_index().groupby("trait").mean().reset_index().drop(columns="fold").to_excel(
        os.path.join(root, "final_performances_coherence.xlsx"), index=False
    )

    if test1:
        df_performance_kfolds.groupby("trait").mean().reset_index().drop(
            columns="fold"
        ).to_excel(os.path.join(root, "final_performances_kfolds.xlsx"), index=False)
