import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def cleaning_text(text: str) -> str:
    ret_text = text

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


def top_documents(list_of_documents: list, how_many: int = 10) -> list:
    working_list = list_of_documents.copy()
    result = []
    for i in range(how_many):
        token_index = np.argmax(working_list)
        result.append(token_index)
        working_list[token_index] = 0
    return result
