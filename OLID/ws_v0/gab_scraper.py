import json

# reading the data from the file
with open('GABPOSTS_2016-09.txt') as f:
	data = [next(f) for x in range(10000)]


comments = []

for d in data:
	js = json.loads(d)
	comments.append(js['body'])


with open('gab_comments.txt', 'w', encoding='utf-8') as f:
	for comment in comments:
		comment = comment.replace('\n', ' ')
		f.write(f'{comment}\n')