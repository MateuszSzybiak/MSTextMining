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
    pretty_table
    )
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
    )


df_true = pd.read_csv(r"D:\Studia\Python\News_dataset\True.csv")
string = ""
for i in tqdm(range(len(df_true['title']))):
    string += df_true['title'].iloc[i] + " "

stemmed_text_true = stemming(remove_stop_words(cleaning_text(string).split()))
bow_true = bag_of_words(stemmed_text_true)

df_fake = pd.read_csv(r"D:\Studia\Python\News_dataset\Fake.csv")
string_fake = ""
for i in tqdm(range(len(df_fake['title']))):
    string_fake += df_fake['title'].iloc[i] + " "

stemmed_text_fake = stemming(remove_stop_words(cleaning_text(string_fake).split()))
bow_fake = bag_of_words(stemmed_text_fake)

only_true = {k: bow_true[k] for k in set(bow_true) - set(bow_fake)}

only_fake = {k: bow_fake[k] for k in set(bow_fake) - set(bow_true)}

pretty_table(top_dict(only_true), 'prawdziwych')
plot(top_dict(only_true), 'prawdziwych')

pretty_table(top_dict(only_fake), 'fałszywych')
plot(top_dict(only_fake),  'fałszywych')
