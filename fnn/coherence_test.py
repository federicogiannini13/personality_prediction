import os
from fnn.config.coherence_test_config import config
from fnn.modules import data_loader, coherence_checker
import pandas as pd
import sys

sys.path.insert(0, "../")

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
base_root = os.path.join(
    config.OUTPUTS_DIR,
    "outputs",
    config.embedding_name,
    "coherence_test",
    str(config.folds_number) + "_folds",
)

for distance in config.distances:
    root = os.path.join(base_root, str(distance) + "_dist")
    dl = data_loader.Data_Loader(
        traits=config.ocean_traits,
        distance=distance,
        embedding_name=config.embedding_name,
        k_folds=config.folds_number,
    )

    for fold in range(0, config.folds_number):
        for cont_tr, trait in enumerate(config.ocean_traits):
            dl.data[cont_tr].initialize_data_kfolds_cv(fold)
            dl.data[cont_tr].search_unknown_neighbors(distance=distance)
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
                    os.path.join(base_root, "performances_kfolds.xlsx"), index=False
                )
            checker.train2_coherence(
                epochs=config.epochs[1],
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
                os.path.join(root, "performances_coherence.xlsx"), index=False
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
                    epochs=config.epochs[1],
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
                    os.path.join(root, "performances_coherence.xlsx"), index=False
                )
                e += config.epochs_interval

    df_performance_coherence.groupby(["trait", "fold"]).max(
        "best_r2"
    ).reset_index().groupby("trait").mean().reset_index().drop(columns="fold").to_excel(
        os.path.join(
            root,
            "final_performances_coherence_"
            + str(config.folds_number)
            + "folds_"
            + str(distance)
            + "dist_"
            + config.embedding_name
            + ".xlsx",
        ),
        index=False,
    )

    if config.test1:
        df_performance_kfolds.groupby("trait").mean().reset_index().drop(
            columns="fold"
        ).to_excel(
            os.path.join(
                root,
                "final_performances_coherence_"
                + str(config.folds_number)
                + "folds_"
                + str(distance)
                + "dist_"
                + config.embedding_name
                + ".xlsx",
            ),
            index=False,
        )
