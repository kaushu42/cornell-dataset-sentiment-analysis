from utils import load_model, preprocess_text
try:
    classifier = load_model('classifier.pickle')
    vectorizer = load_model('vectorizer.pickle')
except:
    print('Error loading the model. Please make sure you have trained the model using train.py first.')
    exit(-1)

review = [input('Enter a review: ')]
review = preprocess_text(review)
x = vectorizer.transform(review).toarray()
print('Negative Review' if classifier.predict(x)[0] == 0 else 'Positive Review')
