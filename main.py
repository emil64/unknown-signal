import utilities as u
import numpy as np
from matplotlib import pyplot as plt


def split_train_test(x_, y_):
    x = x_
    y = y_
    vx = np.array([x[0], x[4], x[9], x[14], x[19]])
    vy = np.array([y[0], y[4], y[9], y[14], y[19]])
    nx = np.delete(x, [0, 4, 9, 14, 19])
    ny = np.delete(y, [0, 4, 9, 14, 19])
    return nx, ny, vx, vy


class Function:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.error = 0
        self.a = np.zeros(len(self.x))

    def solve(self):
        return self.error

    def plot(self):
        self.error = 0


class Poly(Function):

    def solve(self):
        (nx, ny, vx, vy) = split_train_test(self.x, self.y)
        error = 10000000000001
        c_min = np.zeros(len(self.x))
        for i in range(2, len(nx) + 1):
            c = self.least_squares(nx, ny, i)
            if calc_error(vy, self.calc_y_hat(c, vx)) < error:
                error = calc_error(vy, self.calc_y_hat(c, vx))
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
        print(g - 1)
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


class Exp(Function):

    def solve_exp(self):
        self.a = self.least_squares(self.x, self.y)
        return calc_error(self.y, self.calc_y_hat(self.a, self.x))

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
        return a[0] + a[1] * np.sin(x)

    def plot(self):
        x_new = np.linspace(self.x.min(), self.x.max(), 1000)
        y_new = self.calc_y_hat(self.a, x_new)
        ax.plot(x_new, y_new, '-y')


def solve(x, y):
    poly = Poly(x, y)
    poly.solve()

    sin = Sine(x, y)
    error_sin = sin.solve()

    exp = Exp(x, y)
    error_exp = exp.solve_exp()

    threshold = 25

    min_f = poly
    if error_sin < min_f.error and abs(error_sin - min_f.error) > threshold:
        print(abs(error_sin - min_f.error))
        min_f = sin
    elif error_exp < min_f.error and abs(error_exp - min_f.error) > threshold:
        min_f = exp

    error = min_f.error
    min_f.plot()

    return error


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

# todo implement cos, log
