import csv
import numpy as np


# a + bx + cx^2 +...
def poly(p, noise, offset):
    n = 20
    x = np.random.rand(n).reshape((n, 1))*10 + offset
    e = np.random.normal(0, noise, n).reshape((n, 1))
    y = np.polyval(np.flip(p), x) + e
    return x, y


def sin(p, noise, offset):
    n = 20
    x = np.random.rand(n).reshape((n, 1))*10 + offset
    e = np.random.normal(0, noise, n).reshape((n, 1))
    y = p[0] + p[1]*np.sin(x) + e
    return x, y


def cos(p, noise, offset):
    n = 20
    x = np.random.rand(n).reshape((n, 1))*10 + offset
    e = np.random.normal(0, noise, n).reshape((n, 1))
    y = p[0] + p[1]*np.cos(x) + e
    return x, y


def exp(p, noise, offset):
    n = 20
    x = np.random.rand(n).reshape((n, 1))*10 + offset
    e = np.random.normal(0, noise, n).reshape((n, 1))
    y = p[0] + p[1]*np.exp(x) + e
    return x, y


def log(p, noise, offset):
    n = 20
    x = np.random.rand(n).reshape((n, 1))*10 + offset
    e = np.random.normal(0, noise, n).reshape((n, 1))
    y = p[0] + p[1]*np.log(x) + e
    return x, y


def noise1():
    x_poly, y_poly = poly(np.array([2, 0, 1]), 0.5, -5)
    x_sin, y_sin = sin(np.array([2, 5]), 0.5, 5)
    x = np.append(x_poly, x_sin)
    y = np.append(y_poly, y_sin)

    with open('noise1.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar='\\')
        for i in range(0, len(x)):
            writer.writerow([x[i], y[i]])


noise1()
