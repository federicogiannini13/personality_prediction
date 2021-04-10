from fnn import fnn_model
import numpy as np
import pickle
import sys
import os

from utils import create_dir


class CoerenceChecker:
    """
    Class that performs the Coerence test.
    """
    def __init__(
        self,
        inputs,
        outputs,
        inputs_neig=None,
        test_inputs=None,
        test_outputs=None,
        batch_size=32,
        ocean_traits = [0,1,2,3,4],
        test1 = None
    ):
        """
        Parameters
        ----------
        inputs: numpy array
            numpy array with shape (traning_size, n_features) representing training set's known terms.
        outputs: list
            list of the numpy arrays representing personality scores for training set's known terms.
        inputs_neig: numpy array, default=None
            numpy array with shape (n_unknown_terms, n_features) represening unknown terms to which perform inference and train2.
            Use None if you don't want to perform train2.
        test_inputs: numpy array, default=None
            numpy array with shape (test_size, n_features) representing test set's known terms.
            Use None if you want to perform coerence evaluation on training set.
        test_outputs: list, default=None
            list of the five numpy arrays representing personality scores for test set's known terms.
            Use None if you want to perform coerence evaluation on training set.
        batch_size: int, default=32
            training batch_size of models.
        ocean_traits: list, default=[0, 1, 2, 3, 4]
            Personality traits to which perform test. O: 0, C: 1, E: 2, A: 3, N: 4
        test1: bool
            True if you want to estimate performances of M1's training using test_inputs and test_outputs (only valid if test_inputs and test_outputs are not None)

        Attributes
        ----------
        self.inputs: numpy array
            numpy array with shape (traning_size, n_features) representing training set's known terms.
        self.outputs: list
            list of the numpy arrays representing personality scores for training set's known terms.
        self.inputs_neig: numpy array
            numpy array with shape (n_unknown_terms, n_features) represening unknown terms to which perform inference and train2.
            It is None in the case of no train2.
        self.test_inputs: numpy array
            numpy array with shape (test_size, n_features) representing test set's known terms.
            It is equal to self.train_inputs in case of performing coerence evaluation on training set.
        self.test_outputs: list
            list of the five numpy arrays representing personality scores for test set's known terms.
            It is equal to self.train_outputs in case of performing coerence evaluation on training set.
        self.batch_size: int
            training batch_size of models.
        ocean_traits: list
            Personality traits to which perform test. O: 0, C: 1, E: 2, A: 3, N: 4
        """
        self.inputs = inputs.copy()
        self.original_outputs = outputs.copy()
        self.outputs = outputs.copy()
        self.ocean_traits = ocean_traits
        if inputs_neig is not None:
            self.inputs_neig = inputs_neig.copy()
        else:
            self.inputs_neig = None
        if test_inputs is None:
            self.test_inputs = inputs.copy()
            self.test_outputs = outputs.copy()
            self.test1 = False
        else:
            self.test_inputs = test_inputs.copy()
            self.test_outputs = test_outputs.copy()
            if test1 is None:
                self.test1 = True
            else:
                self.test1 = test1
        self.batch_size = batch_size

        self.all_mse_train2 = []
        self.all_r2_train2 = []
        self.best_epochs_train2 = []
        self.mse = []
        self.r2 = []

    def train1_inference(self, epochs=100, root=None, reset_models=True):
        """
        Perform:
         - train1 of coerence test on self.inputs. After each epoch predict scores of test_inputs and evaluate performance.
         - at the end of train1, if self.inputs_neig is not None, inference of self.inputs_neig's scores.

        Parameters
        ----------
        epochs: int, default=100
            training epochs.
        root: str or path, default=None
            path to which save performance and weights of train1. Use None to not save anything.
        reset_models: True
            use True to-reinitialize models' weights.
            
        Attributes
        -------
        self.models: list of FFNModel
            list containing the M1 models of train1, one for each personality trait.
        self.outputs_neig: list
            list containing M1 models' scores predictions on self.inputs_neig after train1.
        self.best_epochs_train1: list
            best performing train1 epochs for each model (considering predictions of M1 models on test set).
        self.best_weights_train1: list
            weights of the best performing train1 epochs for each model (considering predictions of M1 models on test set).
        self.best_mse_train1: list
            mse score of the best performing train1 epochs for each model (considering predictions of M1 models on test set).
        self.best_r2_train1: list
            r2 score of the best performing train1 epochs for each model (considering predictions of M1 models on test set).
        self.all_mse_train1: list
            mse score of all train1 epochs for each model (considering predictions of M1 models on test set).
        self.all_r2_train1: list
            r2 score of all train1 epochs for each model (considering predictions of M1 models on test set).
        """
        self.outputs_neig = []
        self.final_weights_train1 = []
        if root is not None:
            root_temp = os.path.join(root, "temp")
            create_dir(root)
            create_dir(root_temp)
        if self.test1:
            self.best_weights_train1 = []
            self.best_mse_train1 = []
            self.all_mse_train1 = []
            self.all_r2_train1 = []
            self.best_r2_train1 = []
            self.best_epochs_train1 = []
        if reset_models:
            self.models = []
            for _ in range(0, len(self.ocean_traits)):
                self.models.append(
                    fnn_model.FNNModel(input_dim=np.shape(self.inputs)[1])
                )

        for cont_oc, ocean in enumerate(self.ocean_traits):
            if not self.test1:
                self.models[cont_oc].fit_predict(
                    train_inputs=self.inputs,
                    train_outputs=self.outputs[cont_oc],
                    epochs=epochs,
                    batch_size=self.batch_size,
                )
            else:
                self.models[cont_oc].fit_predict(
                    train_inputs=self.inputs,
                    train_outputs=self.outputs[cont_oc],
                    test_inputs=self.test_inputs,
                    test_outputs=self.test_outputs[cont_oc],
                    epochs=epochs,
                    batch_size=self.batch_size,
                )
                self.best_weights_train1.append(self.models[cont_oc].best_weights)
                self.all_mse_train1.append(self.models[cont_oc].mse)
                self.all_r2_train1.append(self.models[cont_oc].r2)
                ep = np.argmin(self.models[cont_oc].mse)
                self.best_epochs_train1.append(ep)
                self.best_mse_train1.append(self.models[cont_oc].mse[ep])
                self.best_r2_train1.append(self.models[cont_oc].r2[ep])
                if root is not None:
                    with open(os.path.join(root, "all_mse_train1.pickle"), "wb") as f:
                        pickle.dump(self.all_mse_train1, f)
                    with open(os.path.join(root, "all_r2_train1.pickle"), "wb") as f:
                        pickle.dump(self.all_r2_train1, f)
                    with open(
                        os.path.join(root, "best_epochs_train1.pickle"), "wb"
                    ) as f:
                        pickle.dump(self.best_epochs_train1, f)
                    with open(os.path.join(root, "mse_train1.pickle"), "wb") as f:
                        pickle.dump(self.best_mse_train1, f)
                    with open(os.path.join(root, "r2_train1.pickle"), "wb") as f:
                        pickle.dump(self.best_r2_train1, f)

            self.final_weights_train1.append(self.models[cont_oc].model.get_weights())
            if self.inputs_neig is not None:
                out = self.models[cont_oc].model.predict(self.inputs_neig)
                self.outputs_neig.append(out)
            if root is not None:
                with open(os.path.join(root, "outputs_neig.pickle"), "wb") as f:
                    pickle.dump(self.outputs_neig, f)
                with open(os.path.join(root, "final_weights_train1.pickle"), "wb") as f:
                    pickle.dump(self.final_weights_train1, f)

                f = open(
                    os.path.join(root_temp, "0_train_" + str(ocean) + "trait.txt"), "w"
                )
                f.close()

        if self.inputs_neig is not None:
            self.outputs_neig = np.asarray(self.outputs_neig, dtype=np.float32)
        if root is not None:
            with open(os.path.join(root, "outputs_neig.pickle".strip()), "wb") as f:
                pickle.dump(self.outputs_neig, f)

    def train2_coerence(self, epochs=100, root=None):
        """
        Perform:
         - train2 of coerence test on self.inputs_neig.
         - at the end of each train2's epoch, inference of self.test_inputs' scores and performances evaluation.
         
        Parameters
        ----------
        epochs: int
            train2 epochs number.
        root: str or path.
            path to which save performance and weights of train2. Use None to not save anything.

        Attributes
        -------
        self.best_epochs: list
            best performing train2 epochs for each model (considering predictions of M2 models on test set).
        self.best_weights_train2: list
            weights of the best performing train2 epochs for each model (considering predictions of M1 models on test set).
        self.best_mse_train2: list
            mse score of the best performing train2 epochs for each model (considering predictions of M2 models on test set).
        self.best_r2_train1: list
            r2 score of the best performing train2 epochs for each model (considering predictions of M2 models on test set).
        self.all_mse_train1: list
            mse score of all train2 epochs for each model (considering predictions of M2 models on test set).
        self.all_r2_train1: list
            r2 score of all train2 epochs for each model (considering predictions of M2 models on test set).

        """
        self.all_mse_train2 = []
        self.all_r2_train2 = []
        self.best_epochs_train2 = []
        self.best_mse_train2 = []
        self.best_r2_train2 = []
        self.final_weights_train2 = []
        self.best_weights_train2 = []
        root_temp = os.path.join(root, "temp")
        create_dir(root)
        create_dir(root_temp)

        for cont_oc, ocean in enumerate(self.ocean_traits):
            m = fnn_model.FNNModel()
            m.fit_predict(
                train_inputs=self.inputs_neig,
                train_outputs=self.outputs_neig[cont_oc],
                test_inputs=self.test_inputs,
                test_outputs=self.test_outputs[cont_oc],
                epochs=epochs,
                batch_size=self.batch_size,
            )
            self.all_mse_train2.append(m.mse)
            self.all_r2_train2.append(m.r2)
            best_epoch = np.argmin(m.mse)
            self.best_epochs_train2.append(best_epoch)
            self.best_weights_train2.append(m.best_weights)
            self.best_mse_train2.append(m.mse[best_epoch])
            self.best_r2_train2.append(m.r2[best_epoch])
            self.final_weights_train2.append(m.model.get_weights())
            
            if root is not None:
                with open(os.path.join(root, "all_mse.pickle"), "wb") as f:
                    pickle.dump(self.all_mse_train2, f)
                with open(os.path.join(root, "all_r2.pickle"), "wb") as f:
                    pickle.dump(self.all_r2_train2, f)
                with open(os.path.join(root, "best_epochs.pickle"), "wb") as f:
                    pickle.dump(self.best_epochs_train2, f)
                with open(os.path.join(root, "mse.pickle"), "wb") as f:
                    pickle.dump(self.best_mse_train2, f)
                with open(os.path.join(root, "r2.pickle"), "wb") as f:
                    pickle.dump(self.best_r2_train2, f)
                with open(os.path.join(root, "final_weights_train2.pickle"), "wb") as f:
                    pickle.dump(self.final_weights_train2, f)
                with open(os.path.join(root, "best_weights_train2.pickle"), "wb") as f:
                    pickle.dump(self.best_weights_train2, f)
                f = open(os.path.join(root_temp, "1_coerence_" + str(ocean) + ".txt"), "wt")
            f.close()

    def set_outputs_neig(self, weights):
        self.outputs_neig = []
        for cont_oc, ocean in enumerate(self.ocean_traits):
            m = fnn_model.FNNModel(input_dim=np.shape(self.inputs)[0])
            m.model.set_weights(weights[cont_oc])
            pred = m.model.predict(self.inputs_neig)
            self.outputs_neig.append(pred)
        self.outputs_neig = np.asarray(self.outputs_neig)
