import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Conv2D,
    MaxPooling2D,
    Lambda,
    Input,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import math as M
import numpy as np
import sklearn
import sklearn.metrics as skm
import pickle
import os
from utils import create_dir, convert_out


class NLPModel:
    """
    The class representing the CNN model that learns the embedding of a specific personality trait.
    """

    def __init__(
        self,
        train_inputs,
        train_outputs,
        specific=True,
        weights=None,
        voc_dim=60000,
        features_number=200,
        window_size=5,
        filters_number=100,
        hidden_units=50,
        batch_size=32,
        sentence_length=None,
        train_zeros=False,
    ):
        """
        The init method that creates the model. This model can learn an embedding from scratch or can tune a given embedding.

        Parameters
        ----------
        train_inputs: numpy.array
            The numpy.array containing the encoded reviews of training set.
        train_outputs: numpy.array
            If specific==True, the numpy.array with len train_size, containing the reviews target.
            If specific==False, the numpy.array with shape (train_size, number_of_traits), containing the reviews target.
        specific: bool, default: True
            If True the embedding will be tuned considering only one trait.
            If False the embedding will be tuned considering all the traits together.
        weights: list, default: None
            In the case of embedding tuning, this parameter represents the list, with shape
            (voc_dim, embedding_feature_number) representing the initial weights of the embedding to be tuned.
            weights[i] must be the representation in the original embedding of term with index=i.
            If weights is given, the model will tune the embedding.
        voc_dim: int, default: 60000
            In the case of embedding's learning from scratch, this parameter represents the vocabolary size.
        features_number: int, default: 200
            In the case of embedding's learning from scratch, this parameter represents the embedding's features number
        window_size: int, default: 5
            The windows dimension of convolution.
        filters_number: int, default: 100
            The  number of convolution's filters.
        hidden_units: int, default: 50
            The number of units in the hidden layer.
        batch_size: int, default: 32
            The training's batch size.
        sentence_length: int, default: None
            The maximum length of a sentence. If none is set to the length of the longest sentence in training set + 20.
        train_zeros: bool, default: False
            True if you want to train the representation of padding tokens (tokens added to pad each review in such a way that all the reviews have the same lenght).

        Parameters
        ----------
        self.model: tensorflow.keras.models.Sequential
            The model to be trained
        self.train_inputs: numpy.array
            The numpy.array containing the encoded reviews of training set.
        self.train_outputs: numpy.array
            The numpy.array with len train_size, containing the reviews target.
        self.embedding_layer: tensorflow.keras.layers.Embedding
            The model's embedding layer.
        self.conv_layer: tensorflow.keras.layers.Conv2D
            The model's convolutional layer.
        self.pool_layer: tensorflow.keras.layers.MaxPooling2D
            The model's max pool layer.
        self.hidden_layer: tensorflow.keras.layers.Dense
            The model's hidden layer before output layer.
        self.output_layer: tensorflow.keras.layers.Dense
            The model's output layer.
        """
        self.voc_dim = voc_dim
        self.features_number = features_number
        self.window_size = window_size
        self.filters_number = filters_number
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.specific = specific

        self.train_zeros = train_zeros

        if sentence_length is not None:
            self.sentence_length = sentence_length
        else:
            self.sentence_length = self._maxLength(train_inputs)
        self.train_inputs = self._createInputs(train_inputs)

        print("IN SHAPE:", self.train_inputs.shape)

        assert train_outputs is not None
        self.train_outputs = np.asarray(train_outputs)
        print("OUT SHAPE:", self.train_outputs.shape)
        if self.specific:
            self.n_traits = 1
            assert len(self.train_outputs.shape) == 1
        else:
            assert len(self.train_outputs.shape) == 2
            self.n_traits = self.train_outputs.shape[0]
        if self.specific:
            self.train_outputs = self._createOutputs(train_outputs, train_outputs.shape[0])
        else:
            self.train_outputs = self._createOutputs(train_outputs, train_outputs.shape[1])

        if weights is not None:
            self.weights = np.asarray(weights)
            self.voc_dim = self.weights.shape[0]
            self.features_number = self.weights.shape[1]
        else:
            self.weights = np.random.randint(
                low=0, high=100, size=(self.voc_dim, self.features_number)
            )
            self.weights = self.weights / 100

        self._initializeModel()

    def _initializeModel(self):
        self._createModel()
        self._compileModel()

    def _maxLength(self, inputs):
        max_l = 0
        for d in inputs:
            if len(d) > max_l:
                max_l = len(d)
        return max_l + 20

    def _createOutputs(self, x, number):
        return x.reshape(number, 1, 1, self.n_traits)

    def _createInputs(self, inp):
        return pad_sequences(
            inp, maxlen=self.sentence_length, padding="post", truncating="post", value=0
        )

    def _createModel(self):
        if self.train_zeros:
            self._createModel_train_zeros()
        else:
            self._createModel_no_train_zeros()

    def _createModel_train_zeros(self):
        self.model = Sequential()
        self.embedding_layer = Embedding(
            input_dim=self.voc_dim, output_dim=self.features_number, name="emb"
        )
        self.embedding_layer.build((None,))
        self.embedding_layer.set_weights([self.weights])
        self.model.add(self.embedding_layer)

        self.model.add(
            Lambda(lambda t: t[..., None])
        )  # modifica della shape in modo che sia 4d, come richiesto da conv2d

        self.conv_layer = Conv2D(
            filters=self.filters_number,
            kernel_size=(self.window_size, self.features_number),
            strides=1,
            padding="valid",
            name="conv",
        )
        self.model.add(self.conv_layer)

        self.pool_layer = MaxPooling2D(
            pool_size=(self.sentence_length - self.window_size + 1, 1), name="pool"
        )
        self.model.add(self.pool_layer)

        self.hidden_layer = Dense(
            self.hidden_units,
            input_dim=self.filters_number,
            activation=tf.nn.relu,
            name="dense",
        )
        self.model.add(self.hidden_layer)

        self.output_layer = Dense(self.n_traits, activation="linear", name="output")
        self.model.add(self.output_layer)

    def _createModel_no_train_zeros(self):
        self.input_layer = Input(shape=(self.sentence_length,), name="input")
        self.embedding_layer = Embedding(
            input_dim=self.voc_dim, output_dim=self.features_number, name="emb"
        )
        self.embedding_layer.build((None,))
        self.embedding_layer.set_weights([self.weights])
        self.layers_inputs = (self.embedding_layer)(self.input_layer)

        self.lambda_not_equal = Lambda(self._not_equal, name="lambda_not_equal")
        self.layers = (self.lambda_not_equal)(self.layers_inputs)

        self.lambda_layer = Lambda(
            lambda t: t[..., None], name="lambda_shape"
        )  # modifica della shape in modo che sia 4d, come richiesto da conv2d
        self.layers = (self.lambda_layer)(self.layers)

        self.conv_layer = Conv2D(
            filters=self.filters_number,
            kernel_size=(self.window_size, self.features_number),
            strides=1,
            padding="valid",
            name="conv",
        )
        self.layers = (self.conv_layer)(self.layers)

        self.pool_layer = MaxPooling2D(
            pool_size=(self.sentence_length - self.window_size + 1, 1), name="pool"
        )
        self.layers = (self.pool_layer)(self.layers)

        self.hidden_layer = Dense(
            self.hidden_units,
            input_dim=self.filters_number,
            activation=tf.nn.relu,
            name="dense",
        )
        self.layers = (self.hidden_layer)(self.layers)

        self.output_layer = Dense(self.n_traits, activation="linear", name="output")
        self.layers = (self.output_layer)(self.layers)

        self.model = tf.keras.models.Model(inputs=self.input_layer, outputs=self.layers)

    def _not_equal(self, x):
        zeros = tf.constant(0, shape=(self.features_number,), dtype=np.float32)
        not_equal = tf.dtypes.cast(M.not_equal(x, zeros), dtype=np.float32)
        return x * not_equal

    def _compileModel(self):
        self.model.compile(optimizer="adagrad", loss="mse", metrics=["mse"])
        for layer in self.model.layers:
            print(layer.name, end=" ")
            print(layer.output_shape)

    def fit_predict(self, test_inputs, test_outputs, root_path=None, epochs_number=10):
        """
        Fit the model on the training set and, at the end of each epoch, evaluate MSE metrics on test set.
        Store performances and model's weights in the specific path.

        Parameters
        ----------
        test_inputs: numpy.array
            the numpy.array containing the encoded reviews of test set.
        test_outputs: numpy.array
            If the embedding is specific, the numpy.array with len test_size, containing the test set's reviews target.
            If the embedding is unique, the numpy.array with shape (test_size, number_of_traits), test set's reviews
            targets.
        root_path: path, default: None
            the path in which store weights and metrics
        epochs_number: int, default: 10
            train epochs' number.

        Parameters
        -------
        self.predictions: list
            The list containing model's predictions on test set after each epochs.
        self.mse: list
            The list containing model's predictions' estimated MSE on test set after each training epochs.
        self.weights: list
            The list containing model's weights after each training epochs.

        """
        if root_path is not None:
            create_dir(root_path)
        test_inputs = self._createInputs(test_inputs)

        self.predictions = []
        self.mse = []
        self.weights = []
        for i in range(0, epochs_number):
            print("\n________\nEPOCH ", i + 1, "/", epochs_number)
            self.model.fit(
                x=self.train_inputs,
                y=self.train_outputs,
                epochs=1,
                batch_size=self.batch_size,
            )
            self.weights.append(self.embedding_layer.get_weights())
            pred = self.model.predict(test_inputs)
            if self.specific:
                pred = pred.reshape(pred.shape[0])
            else:
                pred = pred.reshape(pred.shape[0], self.n_traits)
            self.predictions.append(pred)
            if not self.specific:
                self.predictions.append(pred)
                mse = []
                for i in range(0, self.n_traits):
                    mse.append(skm.mean_squared_error(test_outputs[i,:], pred[:,i]))
                print("\nTEST RESULTS:\nMSE\n")
                for mse_ in mse:
                    print(mse_)
            else:
                mse = skm.mean_squared_error(test_outputs, pred)
                print("\nTEST RESULTS:\nMSE\n", mse)
            self.mse.append(mse)

            if root_path is not None:
                with open(os.path.join(root_path, "mse.pickle"), "wb") as f:
                    pickle.dump(self.mse, f)
                with open(os.path.join(root_path, "weights.pickle"), "wb") as f:
                    pickle.dump(self.weights, f)

        if self.specific:
            with open(os.path.join(root_path, "embedding.pickle"), "wb") as f:
                pickle.dump(self.weights[np.argmin(self.mse)], f)
        else:
            self.mse = np.asarray(self.mse)
            with open(os.path.join(root_path, "embedding.pickle"), "wb") as f:
                pickle.dump(self.weights[np.argmin(self.mse.mean(axis=1))], f)
