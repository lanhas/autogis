import numpy as np

a = np.arange(4).reshape(2, -1)[np.newaxis, ]
b = np.random.rand(2, 2)[np.newaxis, ]
print(np.vstack((a, b)))
