import numpy as np
from perceptron import Perceptron

X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

p = Perceptron(lr=0.1, epochs=20)
p.fit(X, y)

print(p.predict(X))

X1 = np.array([
    [1,1],
    [1,1],
    [1,0],
    [0,1]
])

print(p.predict(X1))

X2 = np.array([
    [0,1],
    [1,0]
])

print(p.predict(X2))
