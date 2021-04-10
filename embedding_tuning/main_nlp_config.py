ocean_traits = [0, 1, 2, 3, 4]
# OCEAN personality traits to which tune the embedding: O:0, C:1, E:2, A:3, N:4
epochs_number = 10
# NLP model's training epochs
num_reviews = 1500000
# number of reviews to use for training (training set + test set)
voc_dim = 6 * 10 ** 4
# number of terms in the tuned embedding
train_zeros = False
# use True if you want to train weights representing padding's tokens, use False otherwise.
output_type = "mean"
# target of the model: 'mean' or 'sum' of known terms' scores in the review.
shuffle = True
# if True review from yelp dataset will be shuffled before extracting num_reviews reviews.
# if False the first num_reviews of yelp dataset will be extracted.
features_config = [100, int(100 / 2), int(100 / 4)]
# configuration of NLP model's architecture: features, filters and hidden units.
embedding_name = "new_tuned_embedding"
# name of the dir to be created that stores the tuned embedding.
load_reviews_from_scratch = False
# use False if you have already loaded and stored reviews, use True if you want to reload and restore reviews.
tune_embedding = True
# use True to train the model, use False otherwise (eg if you just want to load reviews).
