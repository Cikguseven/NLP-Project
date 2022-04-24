from hatesonar import Sonar

sonar = Sonar()

s = ['help', 'fuck', 'god damn it']

z = [sonar.ping(text=comment)['classes'] for comment in s]

print(z)