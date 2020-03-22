from __future__ import print_function

import sys
import utilities as u
import numpy as np
from matplotlib import pyplot as plt


def split_train_test(x_, y_):
    x = x_
    y = y_
    vx = np.array([x[0], x[10], x[19]])
    vy = np.array([y[0], y[10], y[19]])
    nx = np.delete(x, [0, 10, 19])
    ny = np.delete(y, [0, 10, 19])

    # vx = np.array([x[0], x[4], x[9], x[14], x[19]])
    # vy = np.array([y[0], y[4], y[9], y[14], y[19]])
    # nx = np.delete(x, [0, 4, 9, 14, 19])
    # ny = np.delete(y, [0, 4, 9, 14, 19])

    # nx = x
    # ny = y
    # vx = np.array([])
    # vy = np.array([])

    return nx, ny, vx, vy


class Function:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.error = 0
        self.a = np.zeros(len(self.x))
        self.solve()

    def solve(self):
        return self.error

    def plot(self):
        self.error = 0


class Poly(Function):

    def solve(self):
        (nx, ny, vx, vy) = split_train_test(self.x, self.y)
        error = 10000000000001
        threshold = 1.2
        c_min = np.zeros(len(self.x))
        for i in range(2, 7):
            c = self.least_squares(nx, ny, i)
            if calc_error(self.y, self.calc_y_hat(c, self.x))*threshold < error:
                error = calc_error(self.y, self.calc_y_hat(c, self.x))
                c_min = c
        # plot_poly(c_min, x)
        self.a = c_min
        self.error = calc_error(self.y, self.calc_y_hat(c_min, self.x))
        return self.error

    @staticmethod
    def least_squares(x, y, g):
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

    @staticmethod
    def calc_y_hat(a, x):
        return np.polyval(np.flip(a), x)

    def plot(self):
        g = len(self.a)
        # print(g - 1)
        # print(self.a)
        x_new = np.linspace(self.x.min(), self.x.max(), 1000)
        y_new = np.zeros(len(x_new))
        for i in range(0, g):
            y_new += (self.a[i] * (x_new ** i))
        ax.plot(x_new, y_new, '-r')
        # ax.scatter(x, calc_y_hat_poly(a, x), c='b', s=10)


class Sine(Function):

    def solve(self):
        self.a = self.least_squares(self.x, self.y)
        self.error = calc_error(self.y, self.calc_y_hat(self.a, self.x))
        return self.error

    @staticmethod
    def least_squares(x, y):
        x = x.reshape((len(x), 1))
        x_ = np.ones((len(x), 2))

        x_copy = np.sin(x)
        x_[:, 1] = x_copy[:, 0]

        xt = np.transpose(x_)
        a = np.dot(np.dot(np.linalg.inv(np.dot(xt, x_)), xt), y)
        return a

    @staticmethod
    def calc_y_hat(a, x):
        return a[0] + a[1] * np.sin(x)

    def plot(self):
        x_new = np.linspace(self.x.min(), self.x.max(), 1000)
        y_new = self.calc_y_hat(self.a, x_new)
        ax.plot(x_new, y_new, '-b')


class Cosine(Function):

    def solve(self):
        self.a = self.least_squares(self.x, self.y)
        self.error = calc_error(self.y, self.calc_y_hat(self.a, self.x))
        return self.error

    @staticmethod
    def least_squares(x, y):
        x = x.reshape((len(x), 1))
        x_ = np.ones((len(x), 2))

        x_copy = np.cos(x)
        x_[:, 1] = x_copy[:, 0]

        xt = np.transpose(x_)
        a = np.dot(np.dot(np.linalg.inv(np.dot(xt, x_)), xt), y)
        return a

    @staticmethod
    def calc_y_hat(a, x):
        return a[0] + a[1] * np.cos(x)

    def plot(self):
        x_new = np.linspace(self.x.min(), self.x.max(), 1000)
        y_new = self.calc_y_hat(self.a, x_new)
        ax.plot(x_new, y_new, '-b')


class Exp(Function):

    def solve(self):
        self.a = self.least_squares(self.x, self.y)
        self.error = calc_error(self.y, self.calc_y_hat(self.a, self.x))

    @staticmethod
    def least_squares(x, y):
        x = x.reshape((len(x), 1))
        x_ = np.ones((len(x), 2))

        x_copy = np.exp(x)
        x_[:, 1] = x_copy[:, 0]

        xt = np.transpose(x_)
        a = np.dot(np.dot(np.linalg.inv(np.dot(xt, x_)), xt), y)
        return a

    @staticmethod
    def calc_y_hat(a, x):
        return a[0] + a[1] * np.exp(x)

    def plot(self):
        x_new = np.linspace(self.x.min(), self.x.max(), 1000)
        y_new = self.calc_y_hat(self.a, x_new)
        ax.plot(x_new, y_new, '-y')


class Log(Function):

    def solve(self):

        for a in self.x:
            if a <= 0:
                self.error = 1000000000
                break
        else:
            self.a = self.least_squares(self.x, self.y)
            self.error = calc_error(self.y, self.calc_y_hat(self.a, self.x))

    @staticmethod
    def least_squares(x, y):
        x = x.reshape((len(x), 1))
        x_ = np.ones((len(x), 2))

        x_copy = np.log(x)
        x_[:, 1] = x_copy[:, 0]

        xt = np.transpose(x_)
        a = np.dot(np.dot(np.linalg.inv(np.dot(xt, x_)), xt), y)
        return a

    @staticmethod
    def calc_y_hat(a, x):
        return a[0] + a[1] * np.log(x)

    def plot(self):
        x_new = np.linspace(self.x.min(), self.x.max(), 1000)
        y_new = self.calc_y_hat(self.a, x_new)
        ax.plot(x_new, y_new, '-y')


def solve(x, y):
    poly = Poly(x, y)
    sin = Sine(x, y)
    cos = Cosine(x, y)
    exp = Exp(x, y)
    log = Log(x, y)

    threshold = 0.15

    min_f = poly

    if len(poly.a) > 4:
        threshold = 0
    if sin.error < min_f.error and abs(sin.error - min_f.error) > threshold*min_f.error:
        min_f = sin

    if cos.error < min_f.error and abs(cos.error - min_f.error) > threshold*min_f.error:
        min_f = cos

    if exp.error < min_f.error and abs(exp.error - min_f.error) > threshold*min_f.error:
        min_f = exp

    if log.error < min_f.error and abs(log.error - min_f.error) > threshold*min_f.error:
        min_f = log

    error = min_f.error

    min_f.plot()

    return error


def calc_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)


def split_data(x, y):
    s = 0
    for i in range(0, len(xs), 20):
        e = solve(x[i:i + 20], y[i:i + 20])
        # print(e)
        s += e
    print(s)


fig, ax = plt.subplots()
file = sys.argv[1]
xs, ys = u.load_points_from_file(file)
split_data(xs, ys)
if len(sys.argv) > 2 and sys.argv[2] == '--plot':
    u.view_data_segments(xs, ys)
