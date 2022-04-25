import matplotlib.pyplot as plt
import numpy as np

f1 = []

for p, r in zip(precision, recall):
	f1.append(2 * p * r / (p + r))

threshold = np.array(threshold)
f1 = np.array(f1)

plt.plot(threshold, f1)
plt.show()