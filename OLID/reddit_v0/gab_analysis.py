import gab_raw

raw_text = []

for comment in gab_raw.comments[:]:
	raw_text.append(comment['body'])

print(raw_text)
