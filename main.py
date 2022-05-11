import numpy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

K_B, N = 1.38 * 10 ** (-23), 6.02214076 * 10 ** 23


class MolecularDynamics(object):
    def __init__(self, RadiusCutoff:float = 1,
                 InteractionStrength:float = 1.65 * 10 ** (-21),
                 LengthScale:float = 3.4 * 10 ** (-10),
                 TimeStep:float = 4.5 * 10 ** (-4),
                 BoundaryLength:int = 5,
                 NumberOfParticles:int = 256):

        self._N = NumberOfParticles
        self._RC = RadiusCutoff
        self._EPS = InteractionStrength
        self._SIG = LengthScale
        self._H = TimeStep
        self._L = BoundaryLength

        self.position = numpy.random.random((self._N, 3))

        self.velocity = 30 * numpy.random.random((self._N, 3)) - 15

        self._force = self._update_force()



    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, _p):
        self._pos = _p

    @property
    def velocity(self):
        return self._vel

    @velocity.setter
    def velocity(self, _v):
        self._vel = _v

    @property
    def _force(self):
        return self._for

    @_force.setter
    def _force(self, _f):
        self._for = _f

    def get_kinetic_energy(self):
        energy = 1 / 2 * numpy.sum(numpy.abs(self.velocity))
        return energy

    def get_temperature(self):
        temp = 2 * self.get_kinetic_energy() / (3 * K_B * N)
        return temp

    def _lennard_jones_force(self, r:numpy.ndarray):
        _f = 48 * self._EPS / r * (numpy.power(self._SIG / r, 12) - 1 / 2 * numpy.power(self._SIG / r, 6))
        return _f

    def _update_force(self):
        _force = numpy.empty((self._N, 3))
        for i in range(self._N):
            # difference between two particle axis-wise
            _diff = self.position - self.position[i]
            # delete identical particle
            _diff = numpy.delete(_diff, i, axis=0)

            # absolute distance between two particle
            _r = numpy.sqrt(numpy.sum(numpy.power(_diff, 2), axis=-1))

            # distance unitary vector of each pair
            _rh = numpy.transpose(numpy.transpose(_diff) / _r)

            _force[i] = numpy.sum(self._lennard_jones_force(_rh), axis=0)


        return _force

    def update(self):
        self.position = self.position + self._H * self.velocity + self._H ** 2 * self._force / 2

        _temp1, _temp2 = numpy.greater(self.position, self._L), numpy.less(self.position, - self._L)
        _temp3 = numpy.not_equal(_temp1, _temp2)

        _temp4 = self._update_force()

        self.velocity = self.velocity + self._H * (_temp4 + self._force) / (2 * 6.7 * 10 ** (-26))

        self.velocity = numpy.where(_temp3, -self.velocity, self.velocity)
        # print(h * (_temp + self._force) / (2 * 6.7 * 10 ** (-26)))
        # print(self.velocity)
        self._force = _temp4

        print(self.get_temperature())
        return

def main():
    md = MolecularDynamics()
    # while True:
    #     print(md._velocity)
    #     md.update(4.5 * 10 ** (-12))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = plt.scatter([], [], [], marker='o', color='black')

    l = 5

    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    ax.set_zlim(-l, l)

    def init():
        _pos = numpy.transpose(md.position)
        plot._offsets3d = (_pos[0], _pos[1], _pos[2])

    def update(frame):
        _pos = numpy.transpose(md.position)
        plot._offsets3d = (_pos[0], _pos[1], _pos[2])
        md.update()

    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=1)

    plt.show()

if __name__ == '__main__':
    main()

