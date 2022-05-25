def f(x: bool):
    if x:
        print('a')

z = 3

f(True if z > 2 else False)