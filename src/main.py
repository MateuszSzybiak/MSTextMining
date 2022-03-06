from src.cleaning import (
    cleaning_text,
    remove_stop_words,
    stemming
    )


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
