import utilities as u
import numpy as np
from matplotlib import pyplot as plt


def cross_validation(x_, y_):
    x = x_
    y = y_
    v = [(x[0], y[0]), (x[4], y[4]), (x[9], y[9]), (x[14], y[14]), (x[19], y[19])]
    nx = np.delete(x, [0, 4, 9, 14, 19])
    ny = np.delete(y, [0, 4, 9, 14, 19])
    return nx, ny, v


'''
def solve_poly(x, y):
    (x, y, v) = cross_validation(x, y)
    # print(x, y, v)
'''


def least_squares_poly(x, y, g):
    x = x.reshape((len(x), 1))
    x_ = np.ones((len(x), g))
    x_[:, 1] = x[:, 0]

    x_copy = x.copy()
    for p in range(2, g):
        for i in range(0, len(x)):
            x_copy[i] = x_copy.item(i) * x[i]
        x_[:, p] = x_copy[:, 0]

    xt = np.transpose(x_)
    a = np.dot(np.dot(np.linalg.inv(np.dot(xt, x_)), xt), y)
    return a


def solve(x, y):
    c_min = np.zeros(len(x))
    error = 1000000000
    for i in range(1, 10):
        c = least_squares_poly(x, y, 4)
        if calc_error(y, calc_y_hat_poly(c, x)) < error:
            error = calc_error(y, calc_y_hat_poly(c, x))
            c_min = c
    plot_poly(c_min, x)
    return calc_error(y, calc_y_hat_poly(c_min, x))


def plot_poly(a, x):
    g = len(a)
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = np.zeros(len(x_new))
    for i in range(0, g):
        y_new += (a[i] * (x_new ** i))
    ax.plot(x_new, y_new, '-r')


def calc_y_hat_poly(a, x):
    g = len(a)
    y_hat = np.zeros(len(x))
    for i in range(0, g):
        y_hat += (a[i] * (x ** i))
    return y_hat


def calc_error(y, y_hat):
    return np.sum((y_hat - y) ** 2)


def split_data(x, y):
    s = 0
    for i in range(0, len(xs), 20):
        s += solve(x[i:i + 20], y[i:i + 20])
    print(s)


fig, ax = plt.subplots()
xs, ys = u.load_points_from_file('train_data/noise_2.csv')
split_data(xs, ys)
u.view_data_segments(xs, ys)
