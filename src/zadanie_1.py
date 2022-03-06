import re


text_1 = r"Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"
regex_1 = r"\d+\s"

result_1 = re.sub(regex_1, '', text_1)
print(text_1)
print(result_1 + "\n")


text_2 = r"<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>"
regex_2 = r"<[^>]*>"

result_2 = re.sub(regex_2, '', text_2)
print(text_2)
print(result_2 + "\n")


text_3 = r"Lorem ipsum dolor sit amet, consectetur; adipiscing elit.Sed eget mattis sem. " \
         r"Mauris egestas erat quam, ut faucibus eros congue et. Inblandit, mi eu porta; lobortis, " \
         r"tortor nisl facilisis leo, at tristique augue risuseu risus."
regex_3 = r"[^0-9a-zA-Z ]+"

result_3 = re.sub(regex_3, '', text_3)

print(text_3)
print(result_3 + "\n")
