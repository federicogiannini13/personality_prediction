from embedding_tuning.reviews_loader import read_reviews_loaded
import vocab
from embedding_tuning import nlpmodel, reviews_loader
from embedding_tuning.config.main_nlp_config import *
import os
import sys

sys.path.insert(0, "../")

v = vocab.VocabCreator()

if load_reviews_from_scratch:
    l = reviews_loader.ReviewsLoader(
        dict_emb=v.dict_emb.copy(),
        dict_ocean=v.dict_known.copy(),
        weights=v.weights.copy(),
        voc_dim=voc_dim,
        train=2 / 3,
        num_reviews=num_reviews,
        shuffle=shuffle,
        embedding_name=embedding_name,
    )
else:
    l = read_reviews_loaded(embedding_name)

if tune_embedding:
    for ocean in ocean_traits:
        root_ = os.path.join(l.embedding_path, str(ocean) + " ocean")
        m = nlpmodel.NLPModel(
            weights=l.weights.copy(),
            train_inputs=l.train_inputs,
            train_outputs=l.train_outputs[ocean],
            features_number=features_config[0],
            filters_number=features_config[1],
            hidden_units=features_config[2],
            train_zeros=train_zeros,
        )
        m.fit_predict(
            l.test_inputs,
            l.test_outputs[ocean],
            root_path=root_,
            epochs_number=epochs_number,
        )
