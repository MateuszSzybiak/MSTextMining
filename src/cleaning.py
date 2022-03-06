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


def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    list_of_words = text.split(" ")
    filtered_list = [word for word in list_of_words if word not in stop_words]
    ret_text = " ".join(filtered_list)
    return ret_text


def stemming(text: str) -> list:
    result_list = []
    list_of_words = text.split(" ")
    porter = PorterStemmer()
    for word in list_of_words:
        result_list.append(porter.stem(word))
    return result_list


example = r"   dadah DADVA 12 dada 13 :)  i ;<  daDua12 me daIDba <xp>dad12ad" \
          r" </xp> dau.  ours;    daiusd,   you    13"
# print(example)
# print(cleaning_text(example))
# print(remove_stop_words(cleaning_text(example)))

an_ex = r"Hello my friend ;). What have you done? Are you crazy? 12 cows " \
        r"are here and 7 pigs and you did nothing. Connection, connecting, connected. What a irony :)"

print(an_ex)
print(cleaning_text(an_ex))
print(remove_stop_words(cleaning_text(an_ex)))
print(stemming(remove_stop_words(cleaning_text(an_ex))))
