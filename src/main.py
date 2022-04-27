from src.cleaning import (
    text_tokenizer,
)
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

df_true = pd.read_csv(r"D:\Pliki\Studia II stopnia\Semestr II\Text mining\News _dataset\True.csv")
df_fake = pd.read_csv(r"D:\Pliki\Studia II stopnia\Semestr II\Text mining\News _dataset\Fake.csv")
df_true["dataset"] = 1
df_fake["dataset"] = 0

df_joined = pd.concat([df_true, df_fake])

vectorizer = CountVectorizer(tokenizer=text_tokenizer)
df_transform = vectorizer.fit_transform(df_joined['title'])

x_train, x_test, y_train, y_test = train_test_split(df_transform, df_joined['dataset'], test_size=0.3, random_state=42)

classifiers = [DecisionTreeClassifier(), RandomForestClassifier(),
               LinearSVC(), AdaBoostClassifier(), BaggingClassifier()]
for clf in classifiers:
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(f"{clf} Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 4))
