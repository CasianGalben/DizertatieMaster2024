
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import pickle

def load_and_preprocess_data(train_data_path, test_data_path):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    return train_df, test_df

def prepare_stopwords():
    additional_stopwords = {'the', 'a', 'and', 'is', 'in', 'it', 'this', 'that', 'there', 'can', 'be', 'will', 'with', 'are', 'at', 'by', 'an', 'as', 'to', 'of'}
    stop_words = set(stopwords.words('english')).union(additional_stopwords)
    return stop_words

class AdvancedLemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stop_words = prepare_stopwords()
    
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower()) if t not in self.stop_words and t not in string.punctuation]

def train_naive_bayes(train_df):
    vectorizer = TfidfVectorizer(tokenizer=AdvancedLemmaTokenizer(), ngram_range=(1, 3), max_df=0.5, min_df=10)
    model = MultinomialNB()
    pipeline = make_pipeline(vectorizer, model)
    X_train, X_test, y_train, y_test = train_test_split(train_df['Reviews'], train_df['Rating'], test_size=0.2, random_state=42)
    param_grid = {'multinomialnb__alpha': [0.01, 0.1, 0.5, 1, 2]}
    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train, y_train)
    save_model(grid.best_estimator_) 
    return grid, X_test, y_test

def evaluate_model(grid, X_test, y_test):
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),  
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),  
        "Classification Report": classification_report(y_test, y_pred),  
        "Best Parameters": str(grid.best_params_),  
        "Best CV Score": grid.best_score_  
    }
    return results

def save_model(model, filename='trained_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return filename