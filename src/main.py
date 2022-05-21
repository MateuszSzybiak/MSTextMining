from src.cleaning import (
    text_tokenizer,
    top_tokens,
    top_documents,
    top_dict,
    stemming,
    cleaning_text,
    remove_stop_words,
    bag_of_words,
    plot,
    pretty_table,
    plot_most_important,
    pretty_table_most_important,
    key_plot,
    pretty_table_key,
)
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)


df_fake = pd.read_csv(r"D:\Studia\Python\News_dataset\Fake.csv")
string_fake = ""
for i in tqdm(range(len(df_fake['title']))):
    string_fake += df_fake['title'].iloc[i] + " "

stemmed_text_fake = stemming(remove_stop_words(cleaning_text(string_fake).split()))
bow_fake = bag_of_words(stemmed_text_fake)


# only_fake = {k: bow_fake[k] for k in set(bow_fake) - set(bow_true)}
# pretty_table(top_dict(only_fake), 'fałszywych')
# plot(top_dict(only_fake),  'fałszywych')

# vectorizer = CountVectorizer(tokenizer=text_tokenizer)
# X_transform = vectorizer.fit_transform(df_fake['title'])

vectorizer_tfidf = TfidfVectorizer(tokenizer=text_tokenizer)
transform_tfidf = vectorizer_tfidf.fit_transform(df_fake['title'])
columns = vectorizer_tfidf.get_feature_names_out()
weights = transform_tfidf.toarray().mean(axis=0)

key_plot(columns, weights)
# pretty_table_most_important(top_tokens(X_transform.toarray().sum(axis=0), vectorizer.get_feature_names_out(), 15),
#                             bow_fake, "miary binarnej")
#
# plot_most_important(top_tokens(X_transform.toarray().sum(axis=0), vectorizer.get_feature_names_out(), 15),
#                     bow_fake, "miary binarnej")

# # Top 10 most important tokens
# print("Top 10 most important tokens")
# print(top_tokens(transform_tfidf.toarray().sum(axis=0), vectorizer_tfidf.get_feature_names_out(), 10))
