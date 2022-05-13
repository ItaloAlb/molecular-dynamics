import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.md import MD
import numpy

def main():
    md = MD()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    l = 10

    # a = [x - 0.2 * numpy.random.random() if x > 0 else x + 0.2 * numpy.random.random() for x in range(- l, l)]
    # b = [y - 0.2 * numpy.random.random() if y > 0 else y + 0.2 * numpy.random.random() for y in range(- l, l)]
    #
    # print(a, '\n', b)
    #
    # a, b = numpy.meshgrid(a, b, sparse=False)
    #
    # a = a.flatten()
    # b = b.flatten()
    #
    # print(len(a), len(b))
    #
    # print(a, '\n', b)
    #
    # ax.scatter(a, b)

    def update(frame):
        ax.clear()
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)
        ax.scatter(md.rx, md.ry)
        md.update()
        return

    def init():
        ax.scatter(md.rx, md.ry)
        return

    ani = FuncAnimation(fig, update, init_func=init, interval=1)

    plt.show()


if __name__ == '__main__':
    main()