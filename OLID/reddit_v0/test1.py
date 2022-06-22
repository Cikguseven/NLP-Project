parents = {'Brenda': 'Aaron', 'Claire': 'Brenda', 'David': 'Brenda',
'Elis': 'Claire', 'Freddy': 'Claire', 'Gerene': 'David'}

def convert(p_d):
	c_d = {}
	for key, value in p_d.items():
		if value in c_d:
			c_d[value].append(key)
		else:
			c_d[value] = [key]
		if key not in c_d:
			c_d[key] = []
	return c_d

print(convert(parents))

