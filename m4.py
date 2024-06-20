import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('movie_data.csv', encoding='utf-8')

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='liblinear', C=1, penalty='l2')
lr.fit(X_train_tfidf, y_train)

print("score: ", lr.score(X_test_tfidf, y_test))

# Best parameter set: {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1)}
# CV Accuracy: 0.893
# Test Accuracy: 0.896



