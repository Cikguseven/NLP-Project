with open('filtered_tweets.txt') as f:
	mylist = [line.strip() for line in f]

print(max(mylist, key=len))
  