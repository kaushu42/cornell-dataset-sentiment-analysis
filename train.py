import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from utils import load_data, save_model, preprocess_text

nltk.download('stopwords')

X, y = load_data('dataset/')
    
# Preprocessing the data
corpus = preprocess_text(X)

# Creating a Tfidf models which will use 3000 best words
# min_df = Minimum number of documents a word must appear to be included in the model
# max_df = Maximum number of documents a word can appear in to be included in the model
vectorizer = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.85, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

# Perform train-test split on the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a logistic regression model
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Evaluating the model
pred = classifier.predict(x_test)
print(accuracy_score(y_test, pred))

# Save the model
save_model(classifier, 'classifier.pickle')
save_model(vectorizer, 'vectorizer.pickle')