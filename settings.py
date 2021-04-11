import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = ROOT_DIR

file_known_terms = os.path.join(ROOT_DIR, "data", "known_terms", "known_terms.txt")
file_glove = os.path.join(ROOT_DIR, "data", "glove", "glove.txt")
file_reviews = os.path.join(ROOT_DIR, "data", "yelp_dataset", "yelp_academic_dataset_review.json")