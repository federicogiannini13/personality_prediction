import os
while not os.getcwd().endswith("personality_prediction"):
    os.chdir(os.path.dirname(os.getcwd()))
import vocab
from fnn.modules import preprocessing
import pickle
from settings import ROOT_DIR


class Data_Loader:
    """
    Class that creates preprocessing objects for different personality traits.
    """

    def __init__(
        self,
        traits=[0, 1, 2, 3, 4],
        distance=None,
        max_neigs=None,
        embedding_name="new_tuned_embedding",
        k_folds=None,
        train_prop_holdout=None,
        standardize_holdout=True,
        shuffle=True,
        random_state=42,
        words_to_select=None,
    ):
        """
        The init method that creates preprocessing objects for different personality traits.

        Parameters
        ----------
        traits: list, default: [0, 1, 2, 3, 4]
            OCEAN personality traits: O:0, C:1, E:2, A:3, N:4.
        distance: float, default: None
            In the case of coherence test, the distance to which perform the coherence test. Use None if you are not performing coherence test.
        max_neigs: int, default: None
            Maximum number of unknown neighbors to return in the case of distance>0.
            Use None or 0 if you want to return all possible neighbors in the select distance.
        embedding_name: string, default: 'new_tuned_embeddin'
            The embedding to be used. There must be a directory containing the embedding in data folder.
        k_folds: int, default: None
            The number of folds in case of k-fold cross-validation. Use None if you are not using K-fold cv.
        train_prop_holdout: float, default: None
            The proportion of training set on the known terms' set, in case of Holdout validation. Use None if you are not using K-fold cv.
        standardize_holdout: bool, default: True
            In case of Holdout validation, use True if you want to standardize targets.
        shuffle: bool, default: True
            True if you want to shuffle data before splitting in train and test.
        random_state: int, default: 42
            The seed of shuffling. Use None if you don't want the experiment to be repeatable.
        words_to_select: list, default: None
            If you want to select only a subset of words, set this parameter. Use None otherwise.

        Attributes
        ----------
        self.data: list
            The list of preprocessing objects. In the i-th position is stored the preprocessing object associated with the i-th trait of traits' list.
        """
        v = vocab.VocabCreator(load_embedding=embedding_name == "glove")

        if embedding_name != "glove":
            self.data = []
            embedding_path = os.path.join(ROOT_DIR, "data", embedding_name)
            with open(os.path.join(embedding_path, "dict_emb.pickle"), "rb") as f:
                dict_emb_different = pickle.load(f)
            with open(os.path.join(embedding_path, "words.pickle"), "rb") as f:
                words = pickle.load(f)
            for cont_tr, trait in enumerate(traits):
                with open(
                    os.path.join(
                        embedding_path, str(trait) + " trait", "embedding.pickle"
                    ),
                    "rb",
                ) as f:
                    weights = pickle.load(f)[0].tolist()
                data_ = preprocessing.Preprocessing(
                    dict_emb_different.copy(),
                    v.dict_known.copy(),
                    weights.copy(),
                    shuffle=shuffle,
                    random_state=random_state,
                    k_folds=k_folds,
                    words=words.copy(),
                    train_prop_holdout=train_prop_holdout,
                    standardize_holdout=standardize_holdout,
                    words_to_select=words_to_select,
                )
                data_.initialize_dict_unknown(create_tree=distance is not None)
                if train_prop_holdout is not None:
                    data_.search_unknown_neighbors(
                        distance=distance, max_neigs=max_neigs
                    )
                self.data.append(data_)

        else:
            p = preprocessing.Preprocessing(
                v.dict_emb.copy(),
                v.dict_known.copy(),
                v.weights.copy(),
                shuffle=shuffle,
                random_state=random_state,
                k_folds=k_folds,
                words=v.words_emb.copy(),
                train_prop_holdout=train_prop_holdout,
                standardize_holdout=standardize_holdout,
                words_to_select=words_to_select,
            )
            self.data = [p] * len(traits)
            p.initialize_dict_unknown(create_tree=distance is not None)
            if train_prop_holdout is not None:
                p.search_unknown_neighbors(distance=distance, max_neigs=max_neigs)
