import numpy as np
from time import time

from simpliatmos.core.grid import Mesh
from simpliatmos.core.time_integration import RK3Integrator
from simpliatmos.model.states import State
from simpliatmos.model.initialisation import init_state 
from simpliatmos.model.time import Time
from simpliatmos.io.io import IO


class Model:
    def __init__(self, param):
        param.check()
        self.param = param

        self.mesh = Mesh(param)
        self.state = State(param, self.mesh.shape)
        init_state(self.state, self.mesh, self.param)
        self.integrator = RK3Integrator(param, self.mesh)
        self.time = Time(param)
        self.io = IO(param, self.mesh, self.state, self.time)
        self.diags = []

    def run(self):
        tic = time()
        self.save_to_file()

        while not self.time.finished:
            self.set_dt()
            #print("init-p",(self.state.p[50,100]))
            self.step(1)
            self.progress()
            self.compute_diags()
            self.save_to_file()

        toc = time()
        self.print_perf(toc - tic)

    def step(self, nsteps=1):
        for _ in range(nsteps):
            self.integrator.step(self.state, self.param, self.mesh, self.time)

    def set_dt(self):
        if self.param.dt > 0:
            self.time.dt = self.param.dt
            return

        U = self.state.U
        maxU = np.max(np.abs(U.x)) + np.max(np.abs(U.y)) + 1e-99
        self.time.dt = min(self.param.cfl / maxU, self.param.dtmax)

    def print_perf(self, elapsed):
        print()
        nite = self.time.ite - self.time.ite0
        perf = elapsed / (self.mesh.nx * self.mesh.ny * nite)
        print(f"[INFO] Elapsed: {elapsed:.2f} s | perf: {perf:.2e} s/dof")

    def progress(self):
        if (self.time.ite % self.param.nprint == 0) or self.time.finished:
            msg = [
                f"\rite={self.time.ite}",
                f"{self.time.tostring()}",
                f"dt={self.time.dt:.2g}"
            ]
            print(" ".join(msg), end="")

    def compute_diags(self):
        for diag in self.diags:
            diag()

    def save_to_file(self):
        if self.time.save_to_file:
            self.io.write(self.time)
