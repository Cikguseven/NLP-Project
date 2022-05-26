import re
import comment_filter

x = "ḟ̶̹̎̅̋̂ų̴̢̣͈͚̰͍̻͔̦͎͉̬͓̒̀̀͌́͌̚͠͝c̶̱̝̖̳̿͋̋̎͛̐͋k̶̢͔̰̲̼̯̠̭̜͆̏̊͂̀͐̓̓́̈́̚̚͠ ̸͓̜̙̺͇̬̝̟̽̌̓͌͆́̀͘y̶̝͇̹̟̱͉̒̈͆̀͝͠o̸̡̧̪̱̰̺͙͉͓͕̜͒͛̌̿̿̆̊̅̆͝ṵ̵̪̞͉̦̞̐̉'"


filtered_text = comment_filter.c_filter(
    edmw=True,
    input_list=[x])

# x = re.sub("[^A-Za-z0-9 .,!?'/$&@%+\-\(\)]", ' ', x)

# print(" ".join(x.strip().split()))

print(filtered_text)
