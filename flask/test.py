import re
import comment_filter
import unidecode




x = "ḟ̶̹̎̅̋̂ų̴̢̣͈͚̰͍̻͔̦͎͉̬͓̒̀̀͌́͌̚͠͝c̶̱̝̖̳̿͋̋̎͛̐͋k̶̢͔̰̲̼̯̠̭̜͆̏̊͂̀͐̓̓́̈́̚̚͠ ̸͓̜̙̺͇̬̝̟̽̌̓͌͆́̀͘y̶̝͇̹̟̱͉̒̈͆̀͝͠o̸̡̧̪̱̰̺͙͉͓͕̜͒͛̌̿̿̆̊̅̆͝ṵ̵̪̞͉̦̞̐̉'"

x = unidecode.unidecode(x)
print(type(x))

filtered_text = comment_filter.c_filter(
    edmw=True,
    input_list=[x])

# x = re.sub("[^A-Za-z0-9 .,!?'/$&@%+\-\(\)]", ' ', x)

print(" ".join(filtered_text[0].strip().split()))

print(filtered_text)
