import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from prettytable import PrettyTable


def cleaning_text(text: str) -> str:
    ret_text = text
    
    ret_text = re.sub(r"://", '', ret_text)

    # extract emoticons
    list_of_emoticons = re.findall(r":[^0-9a-zA-Z. ]+|;[^0-9a-zA-Z. ]+", ret_text)

    # remove emoticons
    ret_text = re.sub(r":[^0-9a-zA-Z. ]+|;[^0-9a-zA-Z. ]+", '', ret_text)

    ret_text = ret_text.lower()

    # remove numbers
    ret_text = re.sub(r"\d+\s|\d+", ' ', ret_text)

    # remove html
    ret_text = re.sub(r"<[^>]*>", '', ret_text)  

    # remove punctuation marks
    ret_text = re.sub(r"[^0-9a-zA-Z ]+", '', ret_text)

    ret_text = ret_text.strip()

    # remove additional whitespaces
    ret_text = re.sub(r"\s+", ' ', ret_text)

    for em in list_of_emoticons:
        ret_text += " " + em

    return ret_text


def remove_stop_words(text: list) -> list:
    stop_words = set(stopwords.words('english'))
    list_of_words = text
    return [word for word in list_of_words if word not in stop_words]


def stemming(list_of_words: list) -> list:
    porter = PorterStemmer()
    return [porter.stem(word) for word in list_of_words]


def bag_of_words(words: list) -> dict:
    bow = {}
    for word in words:
        if word not in bow.keys():
            bow[word] = 1
        else:
            bow[word] += 1
    return bow


def text_tokenizer(text: str) -> list:
    text_working = text
    text_working = cleaning_text(text_working)
    text_working_list = text_working.split(" ")
    text_working_list = stemming(text_working_list)
    text_working_list = remove_stop_words(text_working_list)

    return [word for word in text_working_list if len(word) > 3]


def top_tokens(list_of_tokens: list, token_words: list, how_many: int = 10) -> list:
    working_list = list_of_tokens.copy()
    result = []
    for i in range(how_many):
        token_index = np.argmax(working_list)
        result.append(token_words[token_index])
        working_list[token_index] = 0
    return result


def apply_positive_negative(row):
    if row in ["Extremely Positive", "Positive"]:
        return "Positive"
    elif row in ["Extremely Negative", "Negative"]:
        return "Negative"
    return "Neutral"
    
    
def plot_most_important(words: list, bow: dict, title: str):
    keys = words[::-1]
    values = [bow[i] for i in words][::-1]

    y_pos = np.arange(len(keys))

    fig, ax = plt.subplots()

    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos, labels=keys)
    ax.set_title(f"{title}")
    plt.show()
    

def pretty_table_most_important(words: list, bow: dict, title: str):
    result = PrettyTable()

    result.field_names = ["Term", "Count"]
    keys = words
    values = [bow[i] for i in words]
    result.title = f"{title}"
    for i, j in zip(keys, values):
        result.add_row([i, j])
    print(result)


def plot(df):
    fig, ax = plt.subplots(figsize=(10,5))
    counts = list(pd.DataFrame(df["Sentiment"].value_counts())['Sentiment'])

    ax.bar(np.arange(0,len(counts),1), counts, color=['yellowgreen','darkred', "blue", 'orange', 'forestgreen'],
           tick_label=pd.DataFrame(df["Sentiment"].value_counts()).index)
    plt.title("Dystrybucja sentymentu")
    plt.xlabel("Sentyment tweetów")
    plt.ylabel("Ilość tweetów")
    plt.show()
    

def key_plot(columns: list, weights: list):
    highest_weights = np.argpartition(weights, -10)[-10:]
    key_tokens = columns[highest_weights]
    key_weight = weights[highest_weights]
    dframe = pd.DataFrame({"Tokens": key_tokens, "TFIDF": key_weight})
    dframe.sort_values(by=["TFIDF"], inplace=True)
    dframe.plot(kind="barh", x="Tokens", y='TFIDF', title="Kluczowe tokeny według miary TFIDF")
    plt.show()
    
    
def pretty_table_key(columns: list, weights: list):
    result = PrettyTable()
    result.field_names = ["Term", "Weight"]
    highest_weights = np.argpartition(weights, -10)[-10:]
    key_tokens = columns[highest_weights]
    key_weight = weights[highest_weights]
    result.title = "Kluczowe tokeny według miary TFIDF"
    dframe = pd.DataFrame({"Tokens": key_tokens, "TFIDF": key_weight})
    dframe.sort_values(by=["TFIDF"], ascending=False, inplace=True)
    for index, row in dframe.iterrows():
        result.add_row([row["Tokens"], row["TFIDF"]])
    print(result)
    
    
def wordcloud(bow):
    wc = WordCloud()
    wc.generate_from_frequencies(bow)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
