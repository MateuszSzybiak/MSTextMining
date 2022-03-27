from src.cleaning import (
    text_tokenizer,
    # stemming,
    # cleaning_text,
    # remove_stop_words
    )
import pandas as pd
from sklearn.feature_extraction.text import (
    # TfidfVectorizer,
    CountVectorizer
    )


df = pd.read_csv(r"D:\Studia\Python\News_dataset\True.csv")

vectorizer = CountVectorizer(tokenizer=text_tokenizer, binary=True)
X_transform = vectorizer.fit_transform(df['title'])
print(X_transform.toarray())
