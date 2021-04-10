import vocab
from fnn import preprocessing
import pickle
from settings import ROOT_DIR
import os

class Data_Loader:
    def __init__(
        self,
        traits:list=[0, 1, 2, 3, 4],
        distance:float=None,
        embedding_name:str="new_tuned_embedding",
        k_folds:int=None,
        train_prop_holdout:float=None,
        standardize_holdout:bool=True,
        shuffle:bool = True,
        random_state:int= 42
    ):
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
                )
                data_.initialize_dict_unknown(create_tree=distance is not None)
                if train_prop_holdout is not None:
                    data_.search_unknown_neighbors(distance=distance)
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
            )
            self.data = [p] * len(traits)
            p.initialize_dict_unknown(create_tree=distance is not None)
            if train_prop_holdout is not None:
                p.search_unknown_neighbors(distance=distance)
