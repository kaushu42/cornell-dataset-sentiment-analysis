# NLP based sentiment analysis

This project is based on classical NLP approach of sentiment analysis. The data from the cornell sentiment analysis dataset.

### Approach of analysis
1. First of all, the data is loaded.

2. All non-word characters, single word characters are removed from the data. Also, the text is changed to lowercase.

3. The corpus is vectorized using TF-IDF transformation.

4. The data is split into test (20%) and train set(80%).

5. A logistic regression classifier is used to fit the data.

### Running the code
You need to have sklearn and nltk installed.

There are two files:
 `train.py`: It is used to train the model and save the models. The dataset is saved pickled to allow faster loading for large datasets. The classifier and vectorizers are saved as well.
 `test.py`: It is used to test the model by supplying your own input.

 ### Possible Improvements
 * You can tune the parameters in TfidfVectorizer.
 * You can use different kind of pre-processing for the text.
 * Different kind of vectorizers like Word2Vec can be used.
 * Try a different model like SVMs or even neural networks