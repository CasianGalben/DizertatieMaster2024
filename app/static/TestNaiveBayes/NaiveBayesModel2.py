from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


train_data_path = 'SRPtrain.csv'
test_data_path = 'SRPtest.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)


vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7, min_df=5)
model = MultinomialNB(alpha=0.5)


pipeline = make_pipeline(vectorizer, model)


pipeline.fit(train_df['Reviews'], train_df['Rating'])
predictions = pipeline.predict(test_df['Reviews'])
accuracy = accuracy_score(test_df['Rating'], predictions)
conf_matrix = confusion_matrix(test_df['Rating'], predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
