import numpy
import numpy as np
from itertools import combinations

class MD:
    def __init__(self,
                 N_ATOM:int = 16,
                 N_DIM:int = 2,
                 RADIUS_CUTOFF:float = 2 ** (1 / 6),
                 DELTA_T:float = 0.005,
                 **L):

        self.step_count = 0.
        self.time = 0.
        self.u_sum = 0.
        self.vir_sum = 0.

        self.VEL_MAG = 10.
        self.N_ATOM = N_ATOM ** N_DIM
        self.N_DIM = N_DIM
        self.RADIUS_CUTOFF = RADIUS_CUTOFF
        self.RADIUS_THRESHOLD = 0.5
        self.DELTA_T = DELTA_T

        self.L_X, self.L_Y, self.L_Z = L.get("L_X", 20), L.get("L_Y", 20), L.get("L_Z", 20)

        random_state = lambda m, s: 4 * m * numpy.random.random(s) - 2 * m
        random_velocity = lambda m, s: m * numpy.random.random(s) - m / 2

        # Initialize position, velocity and acceleration arrays
        if N_DIM == 3:
            self.rx = numpy.linspace(-self.L_X, self.L_X, N_ATOM)
            self.ry = numpy.linspace(-self.L_Y, self.L_Y, N_ATOM)
            self.rz = numpy.linspace(-self.L_Z, self.L_Z, N_ATOM)


            self.vx = random_velocity(self.VEL_MAG, self.N_ATOM)
            self.ax = np.zeros((self.N_ATOM,))

            self.vy = random_velocity(self.VEL_MAG, self.N_ATOM)
            self.ay = np.zeros((self.N_ATOM,))

            self.vz = random_velocity(self.VEL_MAG, self.N_ATOM)
            self.az = np.zeros((self.N_ATOM,))
        if N_DIM == 2:
            self.rx = numpy.array([])
            self.ry = numpy.array([])

            for _x in range(N_ATOM):
                for _y in range(N_ATOM):
                    self.rx = numpy.append(self.rx, _x + 0.1 * numpy.random.random() - 0.05)
                    self.ry = numpy.append(self.ry, _y + 0.1 * numpy.random.random() - 0.05)

            self.vx = random_velocity(self.VEL_MAG, self.N_ATOM)
            self.ax = np.zeros((self.N_ATOM,))

            self.vy = random_velocity(self.VEL_MAG, self.N_ATOM)
            self.ay = np.zeros((self.N_ATOM,))
        if N_DIM == 1:
            self.rx = random_state(self.L_X, N_ATOM)
            self.vx = random_velocity(self.VEL_MAG, self.N_ATOM)
            self.ax = np.zeros((self.N_ATOM,))


    def update(self):
        self.step_count += 1
        self.time = self.step_count * self.DELTA_T
        self.leapfrog(1)
        self.apply_boundary_condition()
        self.compute_forces()
        self.leapfrog(2)

    @staticmethod
    def _periodic_boundary_condition(dr, l):
        if dr > 0.5 * l:
            return dr - l
        elif dr < - 0.5 * l:
            return dr + l
        else:
            return dr

    def compute_forces(self):
        if self.N_DIM == 3:
            self.ax = numpy.zeros((self.N_ATOM, ))
            self.ay = numpy.zeros((self.N_ATOM, ))
            self.az = numpy.zeros((self.N_ATOM, ))
            _pairs = combinations(range(self.N_ATOM), 2)
            for p in _pairs:
                drx = self.rx[p[0]] - self.rx[p[1]]
                drx = self._periodic_boundary_condition(drx, self.L_X)
                dry = self.ry[p[0]] - self.ry[p[1]]
                dry = self._periodic_boundary_condition(dry, self.L_Y)
                drz = self.rz[p[0]] - self.rz[p[1]]
                drz = self._periodic_boundary_condition(drz, self.L_Z)
                rr = drx * drx + dry * dry + drz * drz
                self.u_sum = 0.
                self.vir_sum = 0.
                if rr < self.RADIUS_CUTOFF:
                    rri = 1. / rr
                    rri3 = rri * rri * rri
                    f = 48. * rri3 * (rri3 - 0.5) * rri
                    self.ax[p[0]] += f * drx
                    self.ay[p[0]] += f * dry
                    self.az[p[0]] += f * drz
                    self.ax[p[1]] -= f * drx
                    self.ay[p[1]] -= f * dry
                    self.az[p[1]] -= f * drz
                    self.u_sum += 4. * rri3 * (rri3 - 1.) + 1.
                    self.vir_sum += f * rr
        if self.N_DIM == 2:
            self.ax = numpy.zeros((self.N_ATOM,))
            self.ay = numpy.zeros((self.N_ATOM,))
            _pairs = combinations(range(self.N_ATOM), 2)
            for p in _pairs:
                drx = self.rx[p[0]] - self.rx[p[1]]
                drx = self._periodic_boundary_condition(drx, self.L_X)
                dry = self.ry[p[0]] - self.ry[p[1]]
                dry = self._periodic_boundary_condition(dry, self.L_Y)
                rr = drx * drx + dry * dry
                self.u_sum = 0.
                self.vir_sum = 0.
                if rr < self.RADIUS_CUTOFF:
                    rri = 1. / rr
                    rri3 = rri * rri * rri
                    f = 48. * rri3 * (rri3 - 0.5) * rri
                    self.ax[p[0]] += f * drx
                    self.ay[p[0]] += f * dry
                    self.ax[p[1]] -= f * drx
                    self.ay[p[1]] -= f * dry
                    self.u_sum += 4. * rri3 * (rri3 - 1.) + 1.
                    self.vir_sum += f * rr
        if self.N_DIM == 1:
            self.ax = numpy.zeros((self.N_ATOM,))
            _pairs = combinations(range(self.N_ATOM), 2)
            for p in _pairs:
                drx = self.rx[p[0]] - self.rx[p[1]]
                drx = self._periodic_boundary_condition(drx, self.L_X)
                rr = drx * drx
                self.u_sum = 0.
                self.vir_sum = 0.
                if rr < self.RADIUS_CUTOFF:
                    rri = 1. / rr
                    rri3 = rri * rri * rri
                    f = 48. * rri3 * (rri3 - 0.5) * rri
                    self.ax[p[0]] += f * drx
                    self.ax[p[1]] -= f * drx
                    self.u_sum += 4. * rri3 * (rri3 - 1.) + 1.
                    self.vir_sum += f * rr

    def leapfrog(self, step):
        if self.N_DIM == 3:
            self.vx = self.vx + 0.5 * self.DELTA_T * self.ax
            self.vy = self.vy + 0.5 * self.DELTA_T * self.ay
            self.vz = self.vz + 0.5 * self.DELTA_T * self.az
            if step == 1:
                self.rx = self.rx + self.DELTA_T * self.vx
                self.ry = self.ry + self.DELTA_T * self.vy
                self.rz = self.rz + self.DELTA_T * self.vz
        if self.N_DIM == 2:
            self.vx = self.vx + 0.5 * self.DELTA_T * self.ax
            self.vy = self.vy + 0.5 * self.DELTA_T * self.ay
            if step == 1:
                self.rx = self.rx + self.DELTA_T * self.vx
                self.ry = self.ry + self.DELTA_T * self.vy
        if self.N_DIM == 1:
            self.vx = self.vx + 0.5 * self.DELTA_T * self.ax
            if step == 1:
                self.rx = self.rx + self.DELTA_T * self.vx

    def apply_boundary_condition(self):
        self.rx = numpy.where(self.rx > 0.5 * self.L_X, self.rx - self.L_X, self.rx)
        self.ry = numpy.where(self.ry > 0.5 * self.L_Y, self.ry - self.L_Y, self.ry)
        self.rz = numpy.where(self.rx > 0.5 * self.L_X, self.rx - self.L_X, self.rx)

        self.rx = numpy.where(self.rx < - 0.5 * self.L_X, self.rx + self.L_X, self.rx)
        self.ry = numpy.where(self.ry < - 0.5 * self.L_Y, self.ry + self.L_Y, self.ry)
        self.rz = numpy.where(self.rx < - 0.5 * self.L_X, self.rx + self.L_X, self.rx)
