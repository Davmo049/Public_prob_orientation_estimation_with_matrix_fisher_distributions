import numpy as np
import loss
import loadlibrary
import matplotlib.pyplot as plt
import matplotlib
from np_math import numdiff

def visualize_forward_function():
    MAX_1 = 40
    hg = lambda v: loadlibrary.hg(MAX_1, 2, 0.5, 2.0, v)
    loghg = lambda v: np.log(hg(v))
    def visualize_v(v, index,axs):
        x = np.arange(0, 20, 0.1)
        y = list(map(lambda xx: loghg(v*xx), x))
        y_est = list(map(lambda xx: loss.numpy_hg_approx2(*v*xx), x))
        ax = axs[index-1]
        if index == 4:
            ax.plot(x, y, 'b', label='exact')
            ax.plot(x, y_est, 'r--', label='approximation')
            ax.legend()
        else:
            ax.plot(x, y, 'b', )
            ax.plot(x, y_est, 'r--')

        ax.set_xlabel('a')
        ax.set_ylabel('F(1/2,2,[{0:.1f}, {2:.1f}, {2:.1f}]\u00D7a)'.format(v[0], v[1], v[2]))
    v1 = np.array([1.0, 1.0, 1.0])
    v2 = np.array([0.0, 0.0, 1.0])
    v3 = np.array([0.0, 0.5, 1.0])
    v4 = np.array([0.3, 0.8, 1.0])

    fig = plt.figure(figsize=(12, 3), dpi=300*4)
    gs = fig.add_gridspec(1, 4, wspace=0.5)
    axs = list(map(lambda i: fig.add_subplot(gs[0, i]), range(4)))
    visualize_v(v1, 1, axs)
    visualize_v(v2, 2, axs)
    visualize_v(v3, 3, axs)
    visualize_v(v4, 4, axs)
    p1 = (0,-0.2)
    p2 = (12,3)
    box = matplotlib.transforms.Bbox(np.stack([p1, p2]))
    fig.savefig('images/forward.pdf', bbox_inches=box)

def visualize_backward_function():
    MAX_1 = 40
    hg = lambda v: loadlibrary.hg(MAX_1, 2, 0.5, 2.0, v)
    loghg = lambda v: np.log(hg(v))
    def visualize_v(v, index, axs):
        x = np.arange(0, 20, 0.1)
        diff3 = list(map(lambda xx: numdiff(xx*v, loghg), x))
        diff3_est = list(map(lambda xx: loss.np_log_hg_approx_backward(xx*v), x))
        c = ['r', 'g', 'b']
        ax = axs[index-1]
        for i in range(3):
            di = list(map(lambda xx: xx[i], diff3))
            di_e = list(map(lambda xx: xx[i], diff3_est))
            color = c[i]
            if i == 2 and index == 4:
                ax.plot(x, di, color, label='exact')
                ax.plot(x, di_e, color + '--', label='approximation')
                ax.legend(loc='upper left')
            else:
                ax.plot(x, di, color)
                ax.plot(x, di_e, color + '--')

        ax.set_xlabel('a')
        ax.set_ylabel('\u2207F(1/2,2,[{0:.1f}, {2:.1f}, {2:.1f}]\u00D7a)'.format(v[0], v[1], v[2]))

    v1 = np.array([1.0, 1.0, 1.0])
    v2 = np.array([0.0, 0.0, 1.0])
    v3 = np.array([0.0, 0.5, 1.0])
    v4 = np.array([0.3, 0.8, 1.0])

    fig = plt.figure(figsize=(12, 3), dpi=300*4)
    gs = fig.add_gridspec(1, 4, wspace=0.5)
    axs = list(map(lambda i: fig.add_subplot(gs[0, i]), range(4)))
    visualize_v(v1, 1, axs)
    visualize_v(v2, 2, axs)
    visualize_v(v3, 3, axs)
    visualize_v(v4, 4, axs)
    p1 = (0,-0.2)
    p2 = (12,3)
    box = matplotlib.transforms.Bbox(np.stack([p1, p2]))
    fig.savefig('images/backward.pdf', bbox_inches=box)


def main():
    # visualize_forward_function()
    visualize_backward_function()


if __name__ == '__main__':
    main()
