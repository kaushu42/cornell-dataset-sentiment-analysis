import pickle
import re

from sklearn.datasets import load_files

def load_data_pickle(file_x, file_y):
    # Unpickling the dataset
    with open(file_x, 'rb') as f:
        X = pickle.load(f)
    with open(file_y, 'rb') as f:
        y = pickle.load(f)
    return X, y

def load_data_normal(path):
    reviews = load_files(path)
    X, y = reviews.data, reviews.target
    return X, y

def save_data_pickle(X, y):
    # Store the data so that it can be accessed later
    with open("X.pickle", "wb") as f:
        pickle.dump(X, f)

    with open("y.pickle", "wb") as f:
        pickle.dump(y, f)

# Try to load the pickled data if it exists
def load_data(path):
    try:
        print("----------Loading pickled data.......")
        X, y = load_data_pickle('X.pickle', 'y.pickle')
        print('----------Pickled data loaded.')
    # Load the data using slower method if the above one fails
    except:
        print('!!!!!!!!!!LOAD FAILED! Resorting to loading using sklearn')
        X, y = load_data_normal(path)
        print('----------Data loaded.')
        print('----------Pickling data....')
        save_data_pickle(X, y)
        print('----------Data pickled.')
    return X, y

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_text(X):
    corpus = []
    for i in range(len(X)):
        # Remove all non words characters and convert to lowercase
        review = re.sub(r'\W', ' ', str(X[i])).lower()

        # Remove all single letter words like I, a, ...
        review = re.sub(r'((^[a-z]\s+)|(\s+[a-z]\s+))', ' ', review)
        # Remove all extra spaces we have introduced
        review = re.sub(r'\s+', ' ', review)
        corpus.append(review.strip())
    return corpus
