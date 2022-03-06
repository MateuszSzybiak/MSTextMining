import re

text = r"Lorem ipsum dolor :) sit amet, consectetur; adipiscing elit. " \
       r"Sed eget mattis sem. ;)Mauris ;( egestas erat quam, :< ut faucibus eros congue :> et. " \
       r"In blandit, mi eu porta;lobortis, tortor :-) nisl facilisis leo, at ;< tristique augue risus eu risus ;-)."
regex = r":[^0-9a-zA-Z. ]+|;[^0-9a-zA-Z. ]+"

result = re.findall(regex, text)

for res in result:
    print(res)
