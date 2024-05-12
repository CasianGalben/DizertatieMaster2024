import pandas as pd
import pickle
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string


class AdvancedLemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower()) if t not in stop_words and t not in string.punctuation]


additional_stopwords = {
    'the', 'a', 'and', 'is', 'in', 'it', 'this', 'that', 'there', 'can', 'be', 'will', 'with', 'are', 'at', 'by', 'an', 'as', 'to', 'of'
}
stop_words = set(stopwords.words('english')).union(additional_stopwords)

with open('trained_naive_bayes_model_2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


new_data_path = 'SRPtest.csv'  
data = pd.read_csv(new_data_path)

X_new = data['Reviews']
y_pred = model.predict(X_new)


print(f'Predictions: {y_pred}')