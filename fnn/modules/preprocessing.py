import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold
import random


class Preprocessing:
    """
    Class that manages data associated with an embedding. It is used to prepare data to be inputted into the model.
    Training and test samples are associated with known terms only.
    """

    def __init__(
        self,
        dict_emb,
        dict_known_scores,
        global_embedding,
        dtype=np.float32,
        k_folds=None,
        words=None,
        shuffle=True,
        train_prop_holdout=None,
        random_state=42,
        standardize_holdout=True,
        words_to_select=None,
    ):
        """
        Init method that initialize embedding dictionaries.
        Parameters
        ----------
        dict_emb: dict
            The dictionary of the embedding that, for each word, represents its index.
        dict_known_scores: dict
            The dictionary of known terms that, for each term, contains a list of the five personality traits' scores.
        global_embedding: list
            The embedding representation. List of size voc_dim, that contains in the i-th position the embedding_size representation of the i-th word in the dict_emb.
        dtype: numpy.dtype, default: numpy.float32
            The numpy dtype of inputs.
        k_folds: int, default: None
            The number of folds in case of k-fold cross-validation. Use None if you are not using K-fold cv.
        words: list, default: None
            The words list of the embedding that stores in the i-th position the i-th word of embedding vocabulary.
        shuffle: bool, default: True
            True if you want to shuffle data before splitting in train and test.
        train_prop_holdout: float, default: None
            The proportion of training set on the known terms' set, in case of Holdout validation. Use None if you are not using K-folds cv.
        random_state: int, default: 42
            The seed of shuffling. Use None if you don't want the experiment to be repeatable.
        standardize_holdout: bool, default: True
            In case of Holdout validation, use True if you want to standardize targets.
        words_to_select: list, default: None
            If you want to select only a subset of words, set this parameter. Use None otherwise.

        Attributes
        ----------
        self.dict_emb: dict
            The dictionary of the embedding that for each word represents its index.
        self.dict_known_scores: dict
            The dictionary of known terms that, for each term, contains a list of the five personality traits' scores.
        self.global_embedding: list
            The embedding representation. List of size voc_dim, that contains in the i-th position the embedding_size representation of the i-th word in the dict_emb.
        self.words: list
            The words list of the embedding that stores in the i-th position the i-th word of embedding vocabulary.
        self.words_known: list
            Ordered list of known terms.
        self.embedding_known: list
            The embedding representation of known_terms as a list of size (len(known_terms)), that contains in the i-th position the embedding_size representation of the i-th word in words_known.
        self.dict_known_pos: dict
            The dictionary of known terms that, for each term, contains its index in words_known.
        """
        self.dict_emb = dict_emb
        self.dict_known_scores = dict_known_scores
        self.global_embedding = global_embedding
        self.words = words
        self.r2 = []
        self.dtype = dtype
        self.tree_known = None
        self.random_state = random_state
        self.shuffle = shuffle

        if words_to_select is not None:
            self._filter_words(words_to_select)

        self._initialize_dicts_known()

        if k_folds is not None:
            self.kf = KFold(
                n_splits=k_folds, shuffle=shuffle, random_state=random_state
            )
            self.train_index_kfolds = []
            self.test_index_kfolds = []
            for train_index, test_index in self.kf.split(self.embedding_known):
                self.train_index_kfolds.append(train_index)
                self.test_index_kfolds.append(test_index)
        if train_prop_holdout is not None:
            if train_prop_holdout < 1:
                self.sep = int(train_prop_holdout * len(self.words_known))
            else:
                self.sep = None
            self.initialize_data_holdout(standardize=standardize_holdout)

        print("____ PREPROCESSING DONE")

    def _filter_words(self, words_to_select):
        print("... Filtering words")
        words = []
        embedding = []
        dict_emb = {}
        cont = 0
        for w in self.words:
            if w in words_to_select:
                dict_emb[w] = cont
                words.append(w)
                embedding.append(self.global_embedding[self.dict_emb[w]])
                cont += 1
        words_known = list(self.dict_known_scores.keys())
        for w in words_known:
            if w not in words_to_select:
                del self.dict_known_scores[w]
        self.words = words
        self.global_embedding = embedding
        self.dict_emb = dict_emb
        print("-> words filtered")

    def _initialize_dicts_known(self):
        self.embedding_known = []
        self.output = []
        self.words_known = []
        for i in range(0, 5):
            self.output.append([])
        self.dict_known_pos = {}
        cont = 0
        for w in self.dict_known_scores:
            if w in self.dict_emb:
                self.words_known.append(w)
                self.embedding_known.append(self.global_embedding[self.dict_emb[w]])
                self.dict_known_pos[w] = cont
                for i in range(0, 5):
                    self.output[i].append(self.dict_known_scores[w][i])
                cont += 1

        self.embedding_known = np.asarray(self.embedding_known)
        self.output = np.asarray(self.output)

    def initialize_data_kfolds_cv(self, fold, standardize=True):
        """
        Initialize data for a round of K-fold CV.

        Parameters
        ----------
        fold: int
            the fold number of the KFCV round starting from 0.
        standardize: bool, default: True
            True if you want to standardize train and test outputs.

        Attributes
        -------
        self.train_inputs: numpy array
            The numpy array with shape (train_size, embedding_size) that contains the embedding representations of training set.
        self.train_outputs: list
            The list of the five numpy arrays representing personaliy scores for the training sample.
        self.test_inputs: numpy array
            The numpy array with shape (test_size, embedding_size) that contains embedding representations of test set.
        self.test_outputs: list
            The list of the five numpy arrays representing personaliy scores for the test set.
        self.train_words: numpy array
            The numpy array containing the words of training set.
        self.test_words: numpy array
            The numpy array containing the words of test set.
        self.train_words_dict: dict
            The dictionary containing, for each word in the training set, its position in train_inputs and train_outputs.
        self.train_len: int
            The dimension of training set.
        """
        train_index = self.train_index_kfolds[fold]
        test_index = self.test_index_kfolds[fold]
        self.test_inputs = np.take(self.embedding_known, test_index, axis=0)
        self.test_words = np.take(self.words_known, test_index)
        self.test_outputs = []
        for i in range(0, 5):
            self.test_outputs.append(np.take(self.output[i], test_index, axis=0))
        self.test_outputs = np.asarray(self.test_outputs)

        self.train_inputs = np.take(self.embedding_known, train_index, axis=0)
        self.train_words = np.take(self.words_known, train_index).tolist()
        self.train_outputs = []
        for i in range(0, 5):
            self.train_outputs.append(np.take(self.output[i], train_index, axis=0))
        self.train_outputs = np.asarray(self.train_outputs)

        cont = 0
        self.train_words_dict = {}
        for w in self.train_words:
            self.train_words_dict[w] = cont
            cont += 1

        if standardize:
            self.standardize(True)

        self.train_len = len(self.train_words)

    def initialize_data_holdout(self, standardize=True):
        """
        Initialize data for Holdout validation.

        Parameters
        ----------
        standardize: bool, default: True
            True if you want to standardize train and test outputs.

        Attributes
        -------
        self.train_inputs: numpy.array
            The numpy array with shape (train_size, embedding_size) that contains the embedding representations of training set.
        self.train_outputs: list
            The list of the five numpy arrays representing personality scores for the training set.
        self.test_inputs: numpy.array
            The numpy array with shape (test_size, embedding_size) that contains embedding representations of test set.
        self.test_outputs: list
            The list of the five numpy arrays representing personality scores for the test set.
        self.train_words: numpy.array
            The numpy array containing the words of training set.
        self.test_words: numpy.array
            The numpy array containing the words of test set.
        self.train_words_dict: dict
            The dictionary containing, for each word in the training set, its position in train_inputs and train_outputs.
        self.train_len: int
            The imension of training set.

        """
        embedding_known = np.asarray(self.embedding_known)
        output = np.asarray(self.output)
        embedding_known = np.asarray(embedding_known, dtype=self.dtype)
        output = np.asarray(output, dtype=self.dtype)

        if self.shuffle:
            a = list(range(0, np.shape(self.output)[1]))
            if self.random_state is not None:
                random.Random(self.random_state).shuffle(a)
            else:
                random.shuffle(a)
            self.embedding_known = np.take(self.embedding_known, a, axis=0)
            self.words_known = np.take(self.words_known, a)

            output = []
            for i in range(0, 5):
                output.append(np.take(self.output[i], a, axis=0))
            self.output = np.asarray(output)

            embedding_known = np.asarray(self.embedding_known)
            embedding_known = np.asarray(embedding_known, dtype=self.dtype)

        if not self.sep is None:
            self.train_inputs = embedding_known[0 : self.sep]
            self.train_words = self.words_known[0 : self.sep]
            self.test_inputs = embedding_known[self.sep :]
            self.test_words = self.words_known[0 : self.sep]
            self.train_outputs = []
            self.test_outputs = []
            for i in range(0, 5):
                self.train_outputs.append(output[i][0 : self.sep])
                self.test_outputs.append(output[i][self.sep :])
        else:
            self.train_inputs = embedding_known
            self.train_words = self.words_known
            self.test_inputs = np.array(0)
            self.test_words = []
            self.train_outputs = output
            self.test_outputs = np.array(0)

        cont = 0
        self.train_words_dict = {}
        for w in self.train_words:
            self.train_words_dict[w] = cont
            cont += 1

        if standardize:
            self.standardize(self.sep is not None)

        self.train_len = len(self.train_words)

    def initialize_dict_unknown(self, create_tree=True):
        """
        Initialize dictionaries of unknown terms.

        Parameters
        ----------
        create_tree: bool, default: True
            True if you want to create the tree representation of unknown terms in the embedding.

        Attributes
        -------
        self.embedding_unknown: numpy.array
            The embedding representation of unknown terms. List of size len(unkwnon_terms), that contains in the i-th position the embedding_size representation of the i-th unkwnon term in the dict_unknown.
        self.words_unknown: list
            The ordered list of unknown terms.
        self.dict_unknown: dict
            The dict of unknown terms that for each unknown term contains its index in unknown terms list.
        """
        assert not self.words is None
        print("... Initializing the unknown terms' embedding")
        self.embedding_unknown = list(self.global_embedding.copy())
        canc = []
        self.words_unknown = []
        self.dict_unknown = {}
        for i in range(0, np.shape(self.global_embedding)[0]):
            w = self.words[i]
            if w in self.dict_known_scores:
                canc.append(i)
            else:
                self.words_unknown.append(w)
                self.dict_unknown[w] = len(self.words_unknown) - 1
        canc = np.asarray(canc)
        for i in range(0, canc.shape[0]):
            self.embedding_unknown.pop(canc[i])
            canc = canc - 1
        self.embedding_unknown = np.asarray(self.embedding_unknown)
        if create_tree:
            self.create_unknwon_tree()

    def create_unknwon_tree(self):
        """
        Create the tree representation of unknown terms in the embedding.

        Attributes
        -------
        self.unknown_tree: BallTree
            The tree representation of unknown terms in the embedding.
        """
        print("... Creating the tree")
        self.unknown_tree = BallTree(self.embedding_unknown)

    def search_unknown_neighbors(self, distance=0, max_neigs=None):
        """
        Search unknown neighbors of known terms.

        Parameters
        ----------
        distance: float, default: 0
            The distance within which search neighbors.
            - 0: return only the nearest neighbor of each known term of training set.
            - d>0: return only unknown terms whose distance from any known terms in training set is maximum d.
            - None: return all unknown terms.
        max_neigs: int, default: None
            Maximum number of nieghbors to return in the case of distance>0.
            Use None or 0 if you want to return all possible neighbors in the select distance.

        Attributes
        -------
        self.inputs_neig: list
            The list containing the embedding representations of the found unknown neighbors.
        self.words_neig: list
            The list containing the found unknown neighbors.
        """
        if distance == 0:
            print("Querying the tree...")
            self.neighbors_query = self.unknown_tree.query(
                self.train_inputs[0 : self.train_len],
                k=self.train_len - 1,
                return_distance=True,
                sort_results=True,
            )
            self._search_unknown_neighbors_subjobs()
            self.max_distance = np.max(
                [self.neig_of_neig[n][0] for n in self.neig_of_neig.keys()]
            )
        elif distance is None:
            self.inputs_neig = np.asarray(self.embedding_unknown, dtype=self.dtype)
        else:
            self.neighbors_query = self.unknown_tree.query_radius(
                self.train_inputs[0 : self.train_len], r=distance, return_distance=True
            )
            print("Creating neighbors list...")
            unknown_neigs_distances_dict = {}
            unknown_neigs = []
            for i in range(0, len(self.neighbors_query[1])):
                for j in range(0, len(self.neighbors_query[0][i])):
                    n = self.neighbors_query[0][i][j]
                    if n not in unknown_neigs_distances_dict:
                        unknown_neigs.append(n)
                        unknown_neigs_distances_dict[n] = self.neighbors_query[1][i][j]
                    else:
                        unknown_neigs_distances_dict[n] = min(
                            self.neighbors_query[1][i][j],
                            unknown_neigs_distances_dict[n],
                        )
            unknown_neigs_distances = np.asarray(
                [unknown_neigs_distances_dict[n] for n in unknown_neigs]
            )
            unknown_neigs = np.asarray(unknown_neigs)
            if max_neigs is not None and max_neigs > 0:
                sort_idx = np.argsort(unknown_neigs_distances)
                unknown_neigs = np.take(unknown_neigs, sort_idx)
                unknown_neigs_distances = np.take(unknown_neigs_distances, sort_idx)
                unknown_neigs = unknown_neigs[0:max_neigs]
                unknown_neigs_distances = unknown_neigs_distances[0:max_neigs]
            self.max_distance = np.max(unknown_neigs_distances)
            self.max_distance_neig = np.take(
                unknown_neigs, np.argmax(unknown_neigs_distances)
            )
            self.unknown_neigs = unknown_neigs
            self.unknown_neigs_distances = unknown_neigs_distances
            self.inputs_neig = []
            self.words_neig = []
            for n in unknown_neigs:
                self.inputs_neig.append(self.embedding_unknown[n])
                self.words_neig = self.words_unknown[n]
            self.inputs_neig = np.asarray(self.inputs_neig)

    def _search_unknown_neighbors_subjobs(self):
        self.neighbors_query_dist = self.neighbors_query[0].tolist()
        self.neighbors_query_neig = self.neighbors_query[1].tolist()
        self.neig_of_neig = {}
        self.neighbors = {}

        print("Finding neighbors...")
        for i in range(0, self.train_len):
            self._search_unknown_neighbor(i)

        print("Adding neighbors...")
        self.inputs_neig = []
        self.words_neig = []
        for i in range(0, self.train_len):
            neig_pos = self.neighbors[i]
            neig = self.embedding_unknown[neig_pos]
            self.words_neig.append(self.words_unknown[neig_pos])
            self.inputs_neig.append(neig)
        self.inputs_neig = np.asarray(self.inputs_neig)

    def _search_unknown_neighbor(self, i):
        neig = self.neighbors_query_neig[i][0]
        d = self.neighbors_query_dist[i][0]
        if not neig in self.neig_of_neig:
            self.neig_of_neig[neig] = [d, i]
            self.neighbors[i] = neig
        else:
            if d < self.neig_of_neig[neig][0]:
                other_node = self.neig_of_neig[neig][1]
                self.neighbors_query_neig[other_node].pop(0)
                self.neighbors_query_dist[other_node].pop(0)
                self.neig_of_neig[neig] = [d, i]
                self.neighbors[i] = neig
                self._search_unknown_neighbor(other_node)
            else:
                self.neighbors_query_neig[i].pop(0)
                self.neighbors_query_dist[i].pop(0)
                self._search_unknown_neighbor(i)

    def standardize(self, test=True):
        """
        Standardize train and test outputs.

        Parameters
        ----------
        test: bool, default: True
            True if you want to standardize also test targets.
        """
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(np.transpose(np.asarray(self.train_outputs, dtype=self.dtype)))
        self.train_outputs = np.transpose(
            self.scaler.transform(
                np.transpose(np.asarray(self.train_outputs, dtype=self.dtype))
            )
        )
        if test and len(self.test_outputs[0]) > 0:
            self.test_outputs = np.transpose(
                self.scaler.transform(
                    np.transpose(np.asarray(self.test_outputs, dtype=self.dtype))
                )
            )

    def search_known_nearest_neighbors(self, k=5):
        """
        Search the k known nearest neighbors of all unknown terms.
        Parameters
        ----------
        k: int, default: 5
            numbers of nearest neighbors to be found.

        Attributes
        -------
        self.known_dist: list
            List that contains, in the i-th position, the list of distances between the i-th unknown term in words_unknown and its k nearest known neighbors.
        self.known_neig: list
            List that contains, in the i-th position, the list of k nearest known neighbors' indexes of the i-th unknown term in words_unknown.
        self.words_known_neig: list
            List that contains, in the i-th position, the list of k nearest known neighbors' words of the i-th unknown term in words_unknown.

        """
        print("... Creating the known tree")
        self.tree_known = BallTree(self.embedding_known)
        print("... Querying the known tree")
        self.known_query = self.tree_known.query(
            self.embedding_unknown, k=k, return_distance=True
        )
        self.known_dist = self.known_query[0]
        self.known_neig = self.known_query[1]
        self.words_known_neig = []
        for n in self.known_neig:
            w = np.take(self.words_known, n)
            self.words_known_neig.append(w)

    def is_significant_term(self, i, pred, trait, threshold=0.15, max_dist=10):
        """
        Checks if the i-th unkwnown term (according to words_unknown dictionary) could be significant, using its score prediction.
        To do so, it takes the 5 nearest known terms and calculate their average score.
        The average score is weighted on the inverse distance between known terms and the unknown term, calculated by subtracting max_distance to each distance.
        If the mean is greater than the threshold and its sign is equal to the pred's one, returns True. Otherwise returns False.

        Parameters
        ----------
        i: int
            The index of the word according to the words_unknown dictionary.
        pred: float
            The model's score prediction.
        trait: int
            The personality trait: "O":1, "C":2, "E":3, "A":4, "N":5.
        threshold: float, default: 0.15
            The minimum absolute value for a term to be considered significant.
        max_dist: float, default: 10
            The value to use as maximum distance for inverse distance calculation.

        Returns
        -------
        True if the term could be significant, False otherwise.

        """
        check = False
        if self.tree_known is None:
            self.search_known_nearest_neighbors()
        if np.abs(pred) >= threshold:
            outputs = np.take(self.train_outputs[trait], self.known_neig[i])
            num = 0
            den = 0
            for j in range(0, len(outputs)):
                if np.abs(outputs[j]) >= threshold:
                    num += outputs[j] * (max_dist - self.known_dist[i][j])
                    den += max_dist - self.known_dist[i][j]
            if den > 0:
                avg = num / den
            else:
                avg = 0
            if np.sign(avg) == np.sign(pred) and np.abs(avg) >= threshold:
                check = True
        return check

    def remove_significant(self, i, pred, trait, threshold=0.15, max_dist=10):
        """
        Checks if the i-th unkwnown term (according to words_unknown dictionary) is predicted to be significant and the prediction is consistent, or the term is not predicted to be significant.

        Parameters
        ----------
        i: int
            The index of the word according to the words_unknown dictionary.
        pred: float
            The score prediction of the model.
        trait: int
            The personality trait: "O":1, "C":2, "E":3, "A":4, "N":5.
        threshold: float, default: 0.15
            The minimum absolute value for a term to be considered significant.
        max_dist: float, default: 10
            The value to use as maximum distance for inverse distance calculation.

        Returns
        -------
        True if the term is predicted to be significant and the prediction is consistent, or the term is not predicted to be significant, False otherwise.
        """
        if np.abs(pred) >= threshold:
            return self.is_significant_term(i, pred, trait, threshold, max_dist)
        else:
            return True

    def embedding_test(self, trait, k=3):
        """
        KNN test of the embedding on known terms.


        Parameters
        ----------
        trait: int
            The personality trait: "O":1, "C":2, "E":3, "A":4, "N":5.
        k: int, default: 3
            The number of nearest neighbor of KNN algorithm.

        Parameters
        ----------
        self.known_neigs_known: list
            list containing in the i-th position the K nearest known neighbors' index of the i-th known term.
            Indexes refer to self.dict_known_pos
        self.predictions_knn: list
            list containing in the i-th position the KNN score prediction of the i-th known term.

        Returns
        -------
        r2: float
            The r2 score of KNN's predictions.

        """
        self.tree_known = BallTree(self.train_inputs)
        self.known_neigs_known = self.tree_known.query(
            X=self.train_inputs, k=k + 1, return_distance=False, sort_results=True
        )
        self.predictions_knn = []
        self.outputs_neigs_known = []
        for i in range(0, len(self.known_neigs_known)):
            out = np.take(self.train_outputs[trait], self.known_neigs_known[i][1:])
            self.predictions_knn.append(np.mean(out))
            self.outputs_neigs_known.append(out)
        self.r2_embedding_test = sklearn.metrics.r2_score(
            self.train_outputs[trait], self.predictions_knn
        )
        return self.r2_embedding_test

    def search_neigs_known(self, ocean, k=3, threshold=0.3):
        self.tree_known = BallTree(self.train_inputs[0 : self.train_len])
        self.neigs_known = self.tree_known.query(
            X=self.train_inputs[0 : self.train_len], k=k + 1, return_distance=False
        )
        self.avg_neigs_known = []
        self.outputs_neigs_known = []
        for i in range(0, len(self.neigs_known)):
            out = [
                self.train_outputs[ocean][self.neigs_known[i][j]]
                for j in range(1, len(self.neigs_known[i]))
            ]
            self.avg_neigs_known.append(np.mean(out))
            self.outputs_neigs_known.append(out)
        self.r2.append(
            sklearn.metrics.r2_score(self.train_outputs[ocean], self.avg_neigs_known)
        )
