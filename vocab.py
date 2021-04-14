import csv
from settings import file_glove as file_glove_cfg
from settings import file_known_terms as file_known_terms_cfg


class VocabCreator:
    """
    Class that creates embedding and known vocabularies' objects.
    """

    def __init__(self, file_known_terms=None, file_glove=None, load_embedding=True):
        """
        Init method that creates the vocabularies.

        Parameters
        ----------
        file_known_terms: path
            Path of known_terms.txt file
        file_glove: path
            Path of glove.txt file
        load_embedding: bool
            True if you want to load the embedding. Use True if you are using original GloVe embedding. Use false if you are using tuned embedding

        Attributes
        ----------
        self.dict_known: dict
            dict containing, for each known term, its five personality traits' scores.
        self.dict_emb: dict
            dict containing, for each term in the embedding, its index in the vocabulary.
        self.words_emb: dict
            dict containing for each index in the vocabulary the associated term in the embedding
        self.weights: numpy array
            matrix containg, in the i-th position, the 100-dimensional embedding representation of the i-th term in embedding vocaboluary.
        """

        self.dict_known = {}
        self.dict_emb = {}
        self.words_emb = {}
        self.file_known_terms = file_known_terms_cfg
        self.file_glove = file_glove_cfg
        self.delimiter = " "
        self.weights = []

        if file_known_terms is not None:
            self.file_known_terms = file_known_terms
        if file_glove is not None:
            self.file_glove = file_glove
        self.load_embedding = load_embedding
        self.initialize_embedding()

    def initialize_embedding(self):
        self._read_known_terms()
        if self.load_embedding:
            self._load_embedding()
            self._check_dashes()

    def _read_known_terms(self):
        with open(self.file_known_terms, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                self.dict_known[row[0].lower()] = list(map(float, row[1:6]))
        print("_______ KNOWN VOCABULARY CREATED")

    def _check_dashes(self):
        """
        If a word in dict_emb does not contain dashes, insert the word in dict_known without dashes.
        """
        canc = []
        for w in self.dict_known.keys():
            if "-" in w:
                if not (w in self.dict_emb) and (w.replace("-", "") in self.dict_known):
                    canc.append(w)
        for w in canc:
            x = self.dict_known[w]
            self.dict_known.pop(w)
            self.dict_known[w.replace("-", "")] = x

    def _load_embedding(self):
        with open(self.file_glove, "r", encoding="utf8") as emb:
            reader = csv.reader(emb, delimiter=self.delimiter, quotechar=None)
            cont = 0
            for row in reader:
                self.dict_emb[row[0]] = cont
                self.words_emb[cont] = row[0]
                self.weights.append(
                    list(map(float, row[1 : len(row)]))
                )  # convert string to float
                cont += 1
        # self.dict_emb["-1"]=len(self.dict_emb)
        print("_______ EMBEDDING VOCABULARY CREATED")
