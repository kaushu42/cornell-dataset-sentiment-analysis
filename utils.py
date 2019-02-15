import pickle

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
        print("Loading pickled data.......")
        X, y = load_data_pickle('X.pickle', 'y.pickle')
        print('Pickled data loaded.')
    # Load the data using slower method if the above one fails
    except:
        print('Loading the dataset using sklearn......')
        X, y = load_data_normal(path)
        print('Data loaded.')
        print('Pickling data....')
        save_data_pickle(X, y)
        print('Data pickled.')
    return X, y
