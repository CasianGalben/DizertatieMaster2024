import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the datasets
train_data_path = 'SRPtrain.csv'
test_data_path = 'SRPtest.csv'
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)


train_df.head()
train_df.info()


plt.figure(figsize=(5,5))
sns.countplot(data=train_df, x='Rating', palette=['blue', 'green'])
plt.title("Rating Distribution")
plt.show()


vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7, min_df=5)
model = MultinomialNB(alpha=0.5)


X_train, X_test, y_train, y_test = train_test_split(train_df['Reviews'], train_df['Rating'], test_size=0.2, random_state=42)

pipeline = make_pipeline(vectorizer, model)


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


print(y_pred)


print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification report:\n{classification_report(y_test, y_pred)}")
