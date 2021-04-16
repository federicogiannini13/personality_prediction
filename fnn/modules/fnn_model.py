from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Input,
    LeakyReLU,
    Dropout,
)
import sklearn.metrics
import numpy as np
import pickle
from utils import create_dir


class FNNModel:
    """
    Class that implements the fnn model.
    """

    def __init__(self, input_dim=100, dropout=0.3):
        """
        Init method that builds the model.

        Parameters
        ----------
        input_dim: int, default: 100
            The dimension of the input layer (embedding size).
        dropout: float, default: 0.3
            The value of dropout for the dropout layer.

        Attributes
        ----------
        self.model: Keras Model
            The model to be trained.
        self.input_layer: tensorflow.keras.layers.Input
            The model's input layer.
        self.h1_layer: tensorflow.keras.layers.Dense
            The first model's hidden layer.
        self.h2_layer: tensorflow.keras.layers.Dense
            The second model's hidden layer.
        self.h3_layer: tensorflow.keras.layers.Dense
            The third model's hidden layer.
        self.dropout_layer: tensorflow.keras.layers.Dropout
            The model's dropout layer.
        self.output_layer: tensorflow.keras.layers.Dense
            The model's output layer.
        """
        self.input_dim = input_dim
        self.dropout = dropout
        self.dtype = np.float32
        self.hidden_activation = LeakyReLU(alpha=0.2)
        self.h1_layer_output_dim = 140
        self.h2_layer_output_dim = 70
        self.h3_layer_output_dim = 40
        self._create_layers()
        self._create_model()
        self._compile_model()

    def _create_layers(self):
        self.input_layer = Input(
            shape=(self.input_dim,), name="input", dtype=self.dtype
        )
        self.h1_layer = Dense(
            units=self.h1_layer_output_dim,
            input_dim=self.input_dim,
            activation=self.hidden_activation,
            name="h1",
        )
        self.drop1 = Dropout(self.dropout)
        self.h2_layer = Dense(
            units=self.h2_layer_output_dim,
            input_dim=self.h1_layer_output_dim,
            activation=self.hidden_activation,
            name="h2",
        )
        self.drop2 = Dropout(self.dropout)
        self.h3_layer = Dense(
            units=self.h3_layer_output_dim,
            input_dim=self.h2_layer_output_dim,
            activation=self.hidden_activation,
            name="h3",
        )
        self.dropout_layer = Dropout(self.dropout)
        self.output_layer = Dense(
            units=1, activation=self.hidden_activation, name="out"
        )

    def _create_model(self):
        self.model = Sequential()
        self.model.add(self.h1_layer)
        self.model.add(self.drop1)
        self.model.add(self.h2_layer)
        self.model.add(self.drop2)
        self.model.add(self.h3_layer)
        self.model.add(self.dropout_layer)
        self.model.add(self.output_layer)

    def _compile_model(self):
        self.model.compile(optimizer="adam", loss="mse")

    def fit_predict(
        self,
        train_inputs,
        train_outputs,
        test_inputs=None,
        test_outputs=None,
        epochs=100,
        batch_size=32,
        root=None,
        verbose=2,
        epochs_interval_evaluation=1,
    ):
        """
        Fit on training set and predict on test set.

        Parameters
        ----------
        train_inputs: numpy.array
            The numpy array with shape (training_size, embedding_size) representing the terms to which the model will be trained.
        train_outputs: numpy.array
            The numpy array with len training size containing the targets of training set.
        test_inputs: numpu array, default: None
            The numpy array with shape (test_size, embedding_size) representing the terms of which the model, after each training epoch, will predict scores.
            If None the model will not predict anything.
        test_outputs: numpu array, default: None
            The numpy array with len test size containing the targets of test set.
            If None the model will not predict anything.
        epochs: int, default: 100
            The training epochs.
        batch_size: int, default: 32
            The training batch size.
        root: str or path, default: None.
            The path to which save model's performances.
        verbose: int, default: 2
            The Keras fit method's verbose attribute.
        epochs_interval_evaluation: int, default: 1
            If test set is not None, training will stop after each epochs_interval_evaluation to evaluate performances on test set.

        Attributes
        -------
        self.mse: list
            The mse values evaluated, after each training epoch, on test set's scores predicted by the model.
        self.r2: list
            The r2 scores evaluated, after each training epoch, on test set's scores predicted by the model.
        self.predictions: list
            The test set's scores predicted by the model after each training epoch.
        """
        self.mse = []
        self.r2 = []
        self.predictions = []
        best_mse = 1000
        if root is not None:
            create_dir(root)
        for e in range(0, epochs, epochs_interval_evaluation):
            self.model.fit(
                epochs=epochs_interval_evaluation,
                batch_size=batch_size,
                x=train_inputs,
                y=train_outputs,
                verbose=verbose,
            )
            if test_inputs is not None:
                pred = self.model.predict(test_inputs)
                mse = sklearn.metrics.mean_squared_error(test_outputs, pred)
                r2 = sklearn.metrics.r2_score(test_outputs, pred)
                self.predictions = (
                    self.predictions
                    + [-100] * (epochs_interval_evaluation - 1)
                    + [pred]
                )
                self.mse = self.mse + [100] * (epochs_interval_evaluation - 1) + [mse]
                self.r2 = self.r2 + [-100] * (epochs_interval_evaluation - 1) + [r2]
                print(r2, "\t", mse)
                if mse < best_mse:
                    best_mse = mse
                    self.best_weights = self.model.get_weights()
                    if root is not None:
                        with open(os.path.join(root, "best_weights.pickle"), "wb") as f:
                            pickle.dump(self.best_weights, f)

                if root is not None:
                    with open(os.path.join(root, "mse.pickle"), "wb") as f:
                        pickle.dump(self.mse, f)
                    with open(os.path.join(root, "r2.pickle"), "wb") as f:
                        pickle.dump(self.r2, f)
