import os
import sys
import numpy as np
while not os.getcwd().endswith("personality_prediction") and os.getcwd()!="/":
    os.chdir(os.path.dirname(os.getcwd()))
if os.getcwd()=="/":
    raise Exception("The project dir's name must be 'personality_prediction'. Rename it.")
sys.path.append(os.getcwd())

from embedding_tuning.modules.reviews_loader import read_reviews_loaded
import vocab
from embedding_tuning.modules import nlpmodel, reviews_loader
from embedding_tuning.config.tune_embedding_config import config as config

v = vocab.VocabCreator()

if config.load_reviews_from_scratch:
    l = reviews_loader.ReviewsLoader(
        dict_emb=v.dict_emb.copy(),
        dict_ocean=v.dict_known.copy(),
        weights=v.weights.copy(),
        voc_dim=config.voc_dim,
        train_prop=2 / 3,
        num_reviews=config.num_reviews,
        shuffle=config.shuffle,
        embedding_name=config.embedding_name,
        specific=config.specific
    )
else:
    specific = "_specific" if config.specific else "_unique"
    l = read_reviews_loaded(config.embedding_name+specific)

if config.tune_embedding:
    if config.specific:
        for ocean in config.ocean_traits:
            root_ = os.path.join(l.embedding_path, str(ocean) + " ocean")
            m = nlpmodel.NLPModel(
                specific=True,
                weights=l.weights.copy(),
                train_inputs=l.train_inputs,
                train_outputs=l.train_outputs[ocean],
                features_number=config.features_config[0],
                filters_number=config.features_config[1],
                hidden_units=config.features_config[2],
                train_zeros=config.train_zeros,
            )
            m.fit_predict(
                l.test_inputs,
                l.test_outputs[ocean],
                root_path=root_,
                epochs_number=config.epochs_number,
            )
    else:
        l.train_outputs = np.asarray(l.train_outputs).take(config.ocean_traits, axis=0)
        l.test_outputs = np.asarray(l.test_outputs).take(config.ocean_traits, axis=0)
        root_ = l.embedding_path
        m = nlpmodel.NLPModel(
            specific = False,
            weights=l.weights.copy(),
            train_inputs=l.train_inputs,
            train_outputs=l.train_outputs,
            features_number=config.features_config[0],
            filters_number=config.features_config[1],
            hidden_units=config.features_config[2],
            train_zeros=config.train_zeros,
        )
        m.fit_predict(
            l.test_inputs,
            l.test_outputs,
            root_path=root_,
            epochs_number=config.epochs_number,
        )
