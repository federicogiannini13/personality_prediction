import os
from fnn.main_kfcv_test_config import *
from settings import ROOT_DIR
from fnn import data_loader, fnn_model
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "../")

df_performance = pd.DataFrame(columns=["trait", "fold", "best_epoch", "mse", "r2"])
base_root = os.path.join(
    ROOT_DIR, "outputs", embedding_name, "KFCV", str(folds_number) + "_folds"
)

dl = data_loader.Data_Loader(
    traits=ocean_traits, embedding_name=embedding_name, k_folds=folds_number
)

models = []
for i in range(0, len(ocean_traits)):
    models.append([fnn_model.FNNModel() for i in range(0, folds_number)])

for fold in range(0, folds_number):
    for cont_tr, trait in enumerate(ocean_traits):
        dl.data[cont_tr].initialize_data_kfolds_cv(fold)
        root_ = os.path.join(base_root, str(trait) + "_trait", str(fold) + "_fold")

        m = models[cont_tr][fold]

        m.fit_predict(
            dl.data[cont_tr].train_inputs,
            dl.data[cont_tr].train_outputs[cont_tr],
            dl.data[cont_tr].test_inputs,
            dl.data[cont_tr].test_outputs[cont_tr],
            epochs=epochs,
            batch_size=batch_size,
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
            os.path.join(base_root, "performances.xlsx"), index=False
        )

df_performance.groupby("trait").mean().reset_index().drop(columns="fold").to_excel(
    os.path.join(base_root, "final_performances.xlsx"), index=False
)
