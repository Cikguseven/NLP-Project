import random

def random_seed():
	return 0.3

x = [1, 2, 3, 4]
y = ['a', 'b', 'c', 'd']

random.shuffle(x, random_seed)
random.shuffle(y, random_seed)

print(x)

print(y)