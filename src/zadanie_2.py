import re

text = r"Lorem ipsum dolorsit amet, consectetur adipiscing elit. Sed #texting eget mattis sem. " \
       r"Mauris #frasista egestas erat #tweetext quam, ut faucibus eros #frasier congue et. In blandit, " \
       r"mi eu portalobortis, tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus."
regex = r"#[^\s]*"

result = re.findall(regex, text)

for res in result:
    print(res)
