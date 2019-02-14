import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

nltk.download('stopwords')

def load_data():
    # Unpickling the dataset
    with open('X.pickle', 'rb') as f:
        X = pickle.load(f)
    with open('y.pickle', 'rb') as f:
        y = pickle.load(f)
    return X, y

# Try to load the pickled data if it exists
try:
    X, y = load_data()
    print('Pickled data loaded')
# Load the data using slower method if the above one fails
except:
    print('Loading the dataset using slower method.....')
    reviews = load_files('dataset/')
    X, y = reviews.data, reviews.target

    # Store the data so that it can be accessed later
    with open("X.pickle", "wb") as f:
        pickle.dump(X, f)

    with open("y.pickle", "wb") as f:
        pickle.dump(y, f)

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
