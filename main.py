#import matplotlib.pyplot as plt
import math
import numpy


class MolecularDynamics(object):
    def __init__(self, RadiusCutoff:float = 2,
                 InteractionStrength:float = 1,
                 LengthScale:float = 1,
                 NumberOfParticles:int = 9):

        self._N = NumberOfParticles
        self._RC = RadiusCutoff
        self._EPS = InteractionStrength
        self._SIG = LengthScale



    @property
    def _position(self):
        return self._pos

    @_position.setter
    def _position(self, _p):
        self._pos = _p

    @property
    def _velocity(self):
        return self._vel

    @_velocity.setter
    def _velocity(self, _v):
        self._vel = _v

    @property
    def _force(self):
        return self._for

    @_force.setter
    def _force(self, _f):
        self._for = _f

    def _lennard_jones_force(self, r:numpy.ndarray):
        _f = 48 * self._EPS / r * (numpy.power(self._SIG / r, 12) - 1 / 2 * numpy.power(self._SIG / r, 6))
        return _f

    def _update_force(self):
        _force = numpy.zeros((self._N,), dtype=float)
        for i in range(self._N):
            # difference between two particle axis-wise
            _diff = self._position - self._position[i]
            # delete identical particle
            _diff = numpy.delete(_diff, i, axis=0)

            # absolute distance between two particle
            _r = numpy.sqrt(numpy.sum(numpy.power(_diff, 2), axis=-1))

            # distance unitary vector of each pair
            _rh = numpy.transpose(numpy.transpose(_diff) / _r)

            _force[i] = numpy.sum(self._lennard_jones_force(_rh), axis=0)
        return _force

    def update(self, h):
        self._position = self._position + h * self._velocity + h ** 2 * self._force / 2
        _temp = self._update_force()
        self._velocity = self._velocity + h * [_temp + self._force] / 2
        self._force = _temp
        return

def main():
    md = MolecularDynamics()


if __name__ == '__main__':
    main()

