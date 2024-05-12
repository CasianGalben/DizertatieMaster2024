import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string


train_data_path = 'SRPtrain.csv'
test_data_path = 'SRPtest.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

additional_stopwords = {
    'the', 'a', 'and', 'is', 'in', 'it', 'this', 'that', 'there', 'can', 'be', 'will', 'with', 'are', 'at', 'by', 'an', 'as', 'to', 'of'
}
stop_words = set(stopwords.words('english')).union(additional_stopwords)


class AdvancedLemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower()) if t not in stop_words and t not in string.punctuation]


train_df.head()
train_df.info()


plt.figure(figsize=(5,5))
sns.countplot(data=train_df, x='Rating', palette=['blue', 'green'])
plt.title("Rating Distribution")
plt.show()



vectorizer = TfidfVectorizer(tokenizer=AdvancedLemmaTokenizer(), ngram_range=(1, 3), max_df=0.5, min_df=10)
model = MultinomialNB()

# Creating the pipeline
pipeline = make_pipeline(vectorizer, model)

X_train, X_test, y_train, y_test = train_test_split(train_df['Reviews'], train_df['Rating'], test_size=0.2, random_state=42)

param_grid = {'multinomialnb__alpha': [0.01, 0.1, 0.5, 1, 2]}
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)


print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)


best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(y_pred)


print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification report:\n{classification_report(y_test, y_pred)}")
import pickle

with open('trained_naive_bayes_model_2.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
