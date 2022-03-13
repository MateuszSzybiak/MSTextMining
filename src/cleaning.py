import re
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


def remove_stop_words(text: str) -> list:
    stop_words = set(stopwords.words('english'))
    list_of_words = text.split(" ")
    filtered_list = [word for word in list_of_words if word not in stop_words]
    return filtered_list


def stemming(list_of_words: list) -> list:
    porter = PorterStemmer()
    result_list = [porter.stem(word) for word in list_of_words]
    return result_list


def bag_of_words(words: list) -> dict:
    bow = {}
    for word in words:
        if word not in bow.keys():
            bow[word] = 1
        else:
            bow[word] += 1
    return bow
