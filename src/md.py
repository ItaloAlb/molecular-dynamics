import numpy as np
from itertools import combinations
from math import sqrt


class MolecularDynamic:
    def __init__(self,
                 N_ATOM: int = 121,
                 RADIUS_CUTOFF: float = 2 ** (1 / 6),
                 DELTA_T: float = 0.005,
                 **L):

        self.N_ATOM = N_ATOM
        self.RADIUS_CUTOFF = RADIUS_CUTOFF
        self.DELTA_T = DELTA_T

        self.L_X = L.get("L_X", 20)
        self.L_Y = L.get("L_Y", 20)

        self.rx = np.array([])
        self.ry = np.array([])

        self.initialize_position()

        self.vx = np.array([])
        self.vy = np.array([])

        self.initialize_velocity()

        self.ax = np.array([])
        self.ay = np.array([])

        self.initialize_acceleration()

    def update(self):
        self.leapfrog(1)
        self.apply_boundary_condition()
        self.compute_forces()
        self.leapfrog(2)

    def initialize_position(self):
        sqrt_n_atom: int = int(sqrt(self.N_ATOM))

        self.rx = np.linspace(- self.L_X + 1, self.L_X - 1, sqrt_n_atom)
        self.ry = np.linspace(- self.L_Y + 1, self.L_Y - 1, sqrt_n_atom)

        self.rx, self.ry = np.meshgrid(self.rx, self.ry)

        self.rx = self.rx.flatten()
        self.ry = self.ry.flatten()

        noise = np.random.uniform(-0.125, 0.125, self.N_ATOM)
        self.rx += noise
        noise = np.random.uniform(-0.125, 0.125, self.N_ATOM)
        self.ry += noise

    def initialize_velocity(self):
        self.vx = np.zeros((self.N_ATOM,))
        self.vy = np.zeros((self.N_ATOM,))

    def initialize_acceleration(self):
        self.ax = np.zeros((self.N_ATOM,))
        self.ay = np.zeros((self.N_ATOM,))

    def leapfrog(self, step: int = 2):
        self.vx = self.vx + 0.5 * self.DELTA_T * self.ax
        self.vy = self.vy + 0.5 * self.DELTA_T * self.ay
        if step == 1:
            self.rx = self.rx + self.DELTA_T * self.vx
            self.ry = self.ry + self.DELTA_T * self.vy

    def compute_forces(self):
        self.ax = np.zeros((self.N_ATOM,))
        self.ay = np.zeros((self.N_ATOM,))

        pairs = combinations(range(self.N_ATOM), 2)

        for pair in pairs:
            drx = self.rx[pair[0]] - self.rx[pair[1]]
            drx = self.periodic_boundary_condition(drx, self.L_X)

            dry = self.ry[pair[0]] - self.ry[pair[1]]
            dry = self.periodic_boundary_condition(dry, self.L_Y)

            rr = drx * drx + dry * dry

            if rr < self.RADIUS_CUTOFF:
                rri = 1. / rr
                rri3 = rri * rri * rri
                f = 48. * rri3 * (rri3 - 0.5) * rri
                self.ax[pair[0]] += f * drx
                self.ay[pair[0]] += f * dry

                self.ax[pair[1]] -= f * drx
                self.ay[pair[1]] -= f * dry

    def apply_boundary_condition(self):
        self.rx = np.where(self.rx > self.L_X, self.rx - 2 * self.L_X, self.rx)
        self.ry = np.where(self.ry > self.L_Y, self.ry - 2 * self.L_Y, self.ry)

        self.rx = np.where(self.rx <= - self.L_X, self.rx + 2 * self.L_X, self.rx)
        self.ry = np.where(self.ry <= - self.L_Y, self.ry + 2 * self.L_Y, self.ry)

    def periodic_boundary_condition(self, dr, l):
        if dr > 0.5 * l:
            return dr - l
        elif dr < - 0.5 * l:
            return dr + l
        else:
            return dr