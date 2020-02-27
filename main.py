import utilities as u
import numpy as np
from matplotlib import pyplot as plt


def cross_validation(x_, y_):
    x = x_
    y = y_
    vx = np.array([x[0], x[4], x[9], x[14], x[19]])
    vy = np.array([y[0], y[4], y[9], y[14], y[19]])
    nx = np.delete(x, [0, 4, 9, 14, 19])
    ny = np.delete(y, [0, 4, 9, 14, 19])
    return nx, ny, vx, vy


def solve_poly(x, y):
    (nx, ny, vx, vy) = cross_validation(x, y)
    error = 10000000000001
    c_min = np.zeros(len(x))
    for i in range(2, len(nx)+1):
        c = least_squares_poly(nx, ny, i)
        if calc_error(vy, calc_y_hat_poly(c, vx)) < error:
            error = calc_error(vy, calc_y_hat_poly(c, vx))
            c_min = c
    # plot_poly(c_min, x)
    return calc_error(y, calc_y_hat_poly(c_min, x)), c_min


def solve_sin(x, y):
    c = least_squares_sin(x, y)
    return calc_error(y, calc_y_hat_sin(c, x)), c


def solve_exp(x, y):
    c = least_squares_exp(x, y)
    return calc_error(y, calc_y_hat_exp(c, x)), c


def least_squares_exp(x, y):
    x = x.reshape((len(x), 1))
    x_ = np.ones((len(x), 2))

    x_copy = np.exp(x)
    x_[:, 1] = x_copy[:, 0]

    xt = np.transpose(x_)
    a = np.dot(np.dot(np.linalg.inv(np.dot(xt, x_)), xt), y)
    return a


def least_squares_sin(x, y):
    x = x.reshape((len(x), 1))
    x_ = np.ones((len(x), 2))

    x_copy = np.sin(x)
    x_[:, 1] = x_copy[:, 0]

    xt = np.transpose(x_)
    a = np.dot(np.dot(np.linalg.inv(np.dot(xt, x_)), xt), y)
    return a


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

    (error_poly, a_poly) = solve_poly(x, y)
    (error_sin, a_sin) = solve_sin(x, y)
    (error_exp, a_exp) = solve_exp(x, y)
    if error_poly < error_sin and error_poly < error_exp:
        error = error_poly
        plot_poly(a_poly, x)
    elif error_sin < error_poly and error_sin < error_exp:
        print('sin')
        plot_sin(a_sin, x)
        error = error_sin
    else:
        print('exp')
        plot_exp(a_exp, x)
        error = error_exp

    return error


def plot_poly(a, x):
    g = len(a)
    print(g-1)
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = np.zeros(len(x_new))
    for i in range(0, g):
        y_new += (a[i] * (x_new ** i))
    ax.plot(x_new, y_new, '-r')
    # ax.scatter(x, calc_y_hat_poly(a, x), c='b', s=10)


def plot_sin(a, x):
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = calc_y_hat_sin(a, x_new)
    ax.plot(x_new, y_new, '-b')


def plot_exp(a, x):
    x_new = np.linspace(x.min(), x.max(), 1000)
    y_new = calc_y_hat_exp(a, x_new)
    ax.plot(x_new, y_new, '-y')


def calc_y_hat_poly(a, x):
    return np.polyval(np.flip(a), x)


def calc_y_hat_sin(a, x):
    return a[0] + a[1]*np.sin(x)


def calc_y_hat_exp(a, x):
    return a[0] + a[1] * np.sin(x)


def calc_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)


def split_data(x, y):
    s = 0
    for i in range(0, len(xs), 20):
        e = solve(x[i:i + 20], y[i:i + 20])
        print(e)
        s += e
    print(s)


fig, ax = plt.subplots()
xs, ys = u.load_points_from_file('train_data/adv_3.csv')
split_data(xs, ys)
u.view_data_segments(xs, ys)
