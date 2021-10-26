from utils import create_dir
from embedding_tuning.modules import text_preprocessing
from sklearn.preprocessing import StandardScaler
from settings import file_reviews as file_reviews_cfg
import pickle
import os
from settings import ROOT_DIR
import json
import numpy as np


class ReviewsLoader:
    """
    Class that reads and preprocess Yelp reviews.
    """

    def __init__(
        self,
        dict_emb,
        dict_ocean,
        weights,
        voc_dim=60000,
        train_prop=None,
        file_reviews=None,
        num_reviews=300000,
        save_rev=False,
        shuffle=True,
        output_type="mean",
        embedding_name="new_tuned_embedding",
    ):
        """
        Init method that reads and preprocess Yelp reviews. It follows these steps:
        1) Read the yelp reviews from json file and select num_reviews reviews. Each review is preprocessed with lemmatization and filtering.
        2) Take the most voc_dim frequent words.
        3) Encode each review using words' indexes in dict_emb.
        4) Split the encoded reviews in train and test sets. Standardize the targets of each personality trait.
        5) Write in embedding_path dir the following data: train and test inputs/outputs, the new dict_emb and words, words frequencies and initial wieghtsof embedding.

        Parameters
        ----------
        dict_emb: dict
            The dict containing, for each term in the embedding, its index in the vocabulary.
        dict_ocean: dict
            The dict containing, for each known term, its five personality traits' scores.
        weights: numpy array
            The matrix containg, in the i-th position, the 100-dimensional embedding representation of the i-th term in embedding vocaboluary.
        voc_dim: int, default: 60000
            The desired dimension of tuned embedding vocabulary.
        train_prop: float, default: None
            The train proportion for the CNN model.
        file_reviews: path, default: None
            The path of yelp reviews' json file (yelp_academic_dataset_review.json)
        num_reviews: int, default: 300000
            The number of reviews to be loadead.
        save_rev: bool, default: False
            True if you want to keep the original reviews in reviews attribute.
        shuffle: bool, default: True
            True if you want to shuffle the reviews before reading them. False if you want to take the first num_reviews.
        output_type: str, default: 'mean'
            'mean' if the target of a review, for each personality trait, is the mean of known terms' scores.
            'sum' if the target of a review, for each personality trait, is the sum of known terms' scores.
        embedding_name: str, default: 'new_tuned_embedding'
            The name of the dir to be created that stores the tuned embedding.

        Attributes
        ----------
        dict_emb: dict
            The dict containing, for each term in the embedding, its index in the vocabulary.
        dict_ocean: dict
            The dict containing, for each known term, its five personality traits' scores.
        words: list
            The list containing in the i-th position the word with index i in the embedding vocabulary.
        weights: numpy array
            The matrix containg, in the i-th position, the 100-dimensional embedding representation of the i-th term in embedding vocaboluary.
        train_inputs: numpy array
            The numpy array containing the encoded reviews of the training set.
        test_inputs: numpy array
            The numpy array containing the encoded reviews of the test set.
        train_outputs: numpy array
            The numpy array with shape (5, train_size) containing reviews' scores of training set
        test_outputs: numpy array
            The numpy array with shape (5, test_size) containing reviews' scores of test set
        frequencies: dict
            The dict containing for each word of dict_emb its frequency in the selected reviews.
        """
        self.file_reviews = file_reviews_cfg
        self.reviews = []
        self.reviews_enc = []
        self.labels = []
        self.weights = []
        self.dict_ocean = {}
        self.dict_emb = {}
        self.frequencies = {}

        self.dict_emb = dict_emb
        self.dict_ocean = dict_ocean
        if weights is not None:
            self.weights = weights
        if file_reviews is not None:
            self.file_reviews = file_reviews
        self.num_reviews = num_reviews
        if voc_dim is not None:
            self.voc_dim = voc_dim
        if train_prop is not None:
            self.train_prop = train_prop
            self.sep = int(self.train_prop * self.num_reviews)
        else:
            self.sep = None
        self.embedding_name = embedding_name
        self.embedding_path = os.path.join(ROOT_DIR, "data", embedding_name)
        create_dir(self.embedding_path)

        self.save_rev = save_rev
        self.shuffle = shuffle
        self.output_type = output_type

        self.text_preproc = text_preprocessing.TextPreProcessing()
        self._load_reviews()

    def _load_reviews(self):
        self._read_reviews()
        self._count_frequencies()
        self._update_dict_emb()
        self._encode_reviews()
        self._standardize()
        self._write_data()
        print("______ PREPROCESSING COMPLETED")

    def _read_reviews(self):
        """
        Read the yelp reviews from the json file.
        If self.shuffle=True shuffle reviews before reading them, else take the first self.num_reviews.
        Each review is preprocessed with lemmatiazion and filtering.
        """
        print("reading reviews")
        cont_line = 0
        cont_load = 0
        self.reviews_original = []
        reviews_to_load = 0
        if self.shuffle:
            reviews_to_load = np.asarray(list(range(0, self.num_reviews)))
            np.random.shuffle(reviews_to_load)
            reviews_to_load = reviews_to_load[0 : self.num_reviews]
            reviews_to_load = np.sort(reviews_to_load)
            with open(
                os.path.join(self.embedding_path, "reviews_loaded.pickle"), "wb"
            ) as f:
                pickle.dump(reviews_to_load, f)
        for line in open(self.file_reviews, "r", encoding="utf8"):
            data = json.loads(line)
            if not (self.shuffle) or (cont_line in reviews_to_load):
                rev = data["text"].lower()
                if self.save_rev:
                    self.reviews_original.append(rev)
                rev = self._clear_backspaces(rev)
                rev = self._correct_words(rev)
                self.reviews.append(self.text_preproc.lemmatize(rev))
                cont_load += 1
                if cont_load % 50 == 0:
                    print(cont_load, "/", self.num_reviews, " loaded")
                if self.num_reviews > 0 and cont_load == self.num_reviews:
                    break
            cont_line += 1
        print("_______ REVIEWS READ")

    def _count_reviews(self):
        cont = 0
        for line in open(self.file_reviews, "r", encoding="utf8"):
            print(cont)
        with open(os.path.join(self.embedding_path, "reviews_size.pickle"), "wb") as f:
            pickle.dump(cont, f)

    def _clear_backspaces(self, rev):
        return rev.replace("\n", " ").replace("\t", " ")

    def _correct_words(self, rev):
        return rev.replace("dont", "don't").replace("dont'", "don't")

    def _count_frequencies(self):
        """
        Count frequencies in reviews of the embedding dict's words.
        """
        print("counting frequencies")
        for r in self.reviews[0 : self.sep]:
            for w in r:
                if w in self.dict_emb:
                    if w in self.frequencies:
                        self.frequencies[w] += 1
                    else:
                        self.frequencies[w] = 1
        print("_______ FREQUENCIES COUNTED")

    def _update_dict_emb(self):
        """
        Take only the most voc_dim fequent words.
        Update dict_emb, pos and weights matrix.
        """
        print("Updating dict emb")
        dict_emb = {}
        weights = []
        weights.append(np.zeros(np.shape(self.weights[0])))
        self.words = []
        self.words.append("!")
        dict_emb["!"] = 0
        frequencies_sorted = sorted(
            self.frequencies.items(), reverse=True, key=lambda x: x[1]
        )  # ordinamento decrescente delle frequenze
        end = min(self.voc_dim, np.shape(frequencies_sorted)[0])
        for w in frequencies_sorted[0:end]:
            dict_emb[w[0]] = len(
                dict_emb
            )  # si aggiunge la parola nel nuovo vocabolario
            i = self.dict_emb[w[0]]  # indice della parola nel vecchio vocabalario
            weights.append(
                self.weights[i]
            )  # si aggiungono i pesi della parola nella nuova matrice dei pesi
            self.words.append(w[0])
        self.dict_emb = dict_emb
        self.weights = weights
        print("________ DICT EMB UPDATED")

    def _encode_reviews(self):
        """
        For each review:
        - Encode it using words' indexes in dict_emb (words not present in dict_emb are removed)
        - Calculate, for each personality trait, its target using known terms's scores.
        - Remove reviews with no knonw terms.
        - If self.save_rev=False reset self.reviews value (to avoif out of memory errors).
        """
        print("Encoding reviews")
        i = 0  # tiene il numero della recensione corrente
        self.ocean_words = []
        for r in self.reviews:
            x = []
            lab = np.zeros(5).tolist()
            oc = []
            cont = 0  # conta il numero di parole ocean nella recensione, viene utilizzato come denominatore per la media
            for w in r:
                if w in self.dict_ocean:
                    oc.append(w)
                    for k in range(0, 5):
                        lab[k] += self.dict_ocean[w][k]
                    cont += 1
                if w in self.dict_emb:
                    x.append(self.dict_emb[w])
            if cont != 0:
                if self.output_type == "mean":
                    for i in range(0, len(lab)):
                        lab[i] = lab[i] / cont
                self.labels.append(lab)
                self.reviews_enc.append(x)
                self.ocean_words.append(oc)
            else:
                if i < self.sep:
                    self.sep -= 1
            i += 1
            if i % 50 == 0:
                print(i, "/", len(self.reviews), " encoded")
        if not self.save_rev:
            self.reviews = None
        print("_______ REVIEWS ENCODED")

    def _standardize(self):
        if self.sep is not None:
            self.train_inputs = self.reviews_enc[: self.sep]
            self.test_inputs = self.reviews_enc[self.sep :]
            self.train_outputs = self.labels[: self.sep]
            self.test_outputs = self.labels[self.sep :]
        else:
            self.train_inputs = self.reviews_enc.copy()
            self.train_outputs = self.reviews_enc.copy()
            self.test_inputs = np.asarray(0)
            self.test_outputs = np.asarray(0)
        self.reviews_enc = None
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.train_outputs = self.scaler.fit_transform(self.train_outputs)
        if not self.sep is None:
            self.test_outputs = self.scaler.transform(self.test_outputs)
        self.train_outputs = np.transpose(self.train_outputs)
        self.test_outputs = np.transpose(self.test_outputs)

    def _write_data(self):
        with open(os.path.join(self.embedding_path, "train_inputs.pickle"), "wb") as f:
            pickle.dump(self.train_inputs, f)
        with open(os.path.join(self.embedding_path, "train_outputs.pickle"), "wb") as f:
            pickle.dump(self.train_outputs, f)
        with open(os.path.join(self.embedding_path, "test_inputs.pickle"), "wb") as f:
            pickle.dump(self.test_inputs, f)
        with open(os.path.join(self.embedding_path, "test_outputs.pickle"), "wb") as f:
            pickle.dump(self.test_outputs, f)
        with open(
            os.path.join(self.embedding_path, "weights_initial.pickle"), "wb"
        ) as f:
            pickle.dump(self.weights, f)
        with open(os.path.join(self.embedding_path, "dict_emb.pickle"), "wb") as f:
            pickle.dump(self.dict_emb, f)
        with open(os.path.join(self.embedding_path, "words.pickle"), "wb") as f:
            pickle.dump(self.words, f)
        with open(os.path.join(self.embedding_path, "frequencies.pickle"), "wb") as f:
            pickle.dump(self.frequencies, f)


def read_reviews_loaded(embedding_name="new_tuned_embedding"):
    """
    Read reviews already loaded and preprocessed.

    Parameters
    ----------
    embedding_name: str, default: 'new_tuned_embedding'
        The embedding name corresponding to the directory's name in which are stored its data.

    Returns
    -------
    l : Object
        The object containing read data.

    """

    class ReviewsLoaded:
        pass

    embedding_path = os.path.join(ROOT_DIR, "data", embedding_name)
    l = ReviewsLoaded()
    l.embedding_path = embedding_path
    with open(os.path.join(embedding_path, "train_inputs.pickle"), "rb") as f:
        l.train_inputs = pickle.load(f)
    with open(os.path.join(embedding_path, "train_outputs.pickle"), "rb") as f:
        l.train_outputs = pickle.load(f)
    with open(os.path.join(embedding_path, "test_inputs.pickle"), "rb") as f:
        l.test_inputs = pickle.load(f)
    with open(os.path.join(embedding_path, "test_outputs.pickle"), "rb") as f:
        l.test_outputs = pickle.load(f)
    with open(os.path.join(embedding_path, "weights_initial.pickle"), "rb") as f:
        l.weights = pickle.load(f)
    with open(os.path.join(embedding_path, "dict_emb.pickle"), "rb") as f:
        l.dict_emb = pickle.load(f)
    with open(os.path.join(embedding_path, "words.pickle"), "rb") as f:
        l.words = pickle.load(f)
    with open(os.path.join(embedding_path, "frequencies.pickle"), "rb") as f:
        l.frequencies = pickle.load(f)
    return l
