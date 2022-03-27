from src.cleaning import (
    cleaning_text,
    remove_stop_words,
    stemming,
    bag_of_words
    )
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm


df = pd.read_csv(r"D:\Studia\Python\News_dataset\True.csv")

string = ""
for i in tqdm(range(len(df['title']))):
    string += df['title'].iloc[i] + " "
len(string)

stemmed_text = stemming(remove_stop_words(cleaning_text(string)))

print(bag_of_words(stemmed_text))
bow = bag_of_words(stemmed_text)

wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
