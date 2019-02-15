import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from utils import *

nltk.download('stopwords')

X, y = load_data('dataset/')
# Preprocessing the data
corpus = []

for i in range(len(y)):
    # Remove all non words characters and convert to lowercase
    review = re.sub(r'\W', ' ', str(X[i])).lower()

    # Remove all single letter words like I, a, ...
    review = re.sub(r'((^[a-z]\s+)|(\s+[a-z]\s+))', ' ', review)
    # Remove all extra spaces we have introduced
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

# Creating a BOW models which will use 2000 best words
# min_df = Minimum number of documents a word must appear to be included in the model
# max_df = Maximum number of documents a word can appear in to be included in the model
vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
