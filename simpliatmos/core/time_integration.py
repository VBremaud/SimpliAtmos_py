from simpliatmos.model.states import Prognostic
from simpliatmos.model.equations import get_rhs_and_diag
from simpliatmos.tools.numerics import copyto, addto

class RK3Integrator:
    def __init__(self, param, mesh):
        self.scratch = [Prognostic(param, mesh.shape) for _ in range(3)]
        self.rhs, self.diag = get_rhs_and_diag(param, mesh)

    def step(self, state, param, mesh, time):
        ds1, ds2, ds3 = self.scratch
        #print(type(ds1.u))
        print("initp",state.p[50,100])
        self.rhs(state, ds1)
        addto(state, time.dt, ds1)
        self.diag(state)
        #print(ds1.b[50,100])
        self.rhs(state, ds2)
        addto(state, -3*time.dt/4, ds1, time.dt/4, ds2)
        self.diag(state)
        #print(ds2.b[50,100])
        self.rhs(state, ds3)
        addto(state, -time.dt/12, ds1, -time.dt/12, ds2, 2*time.dt/3, ds3)
        self.diag(state)
        #print(ds3.b[50,100])
        time.pushforward()
