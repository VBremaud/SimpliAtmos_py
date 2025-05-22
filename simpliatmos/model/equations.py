from simpliatmos.tools import operators as op


def get_euler(param, mesh):

    def rhs(s, ds):
        """ RHS for Euler model in momentum-pressure"""
        op.addvortexforce(param, mesh, s.U, s.omega, ds.u)
        op.addgrad(mesh, s.ke, ds.u)
        op.fill(mesh, ds.u)

    def diag(s):
        op.pressure_projection(mesh, s.U, s.div, s.p, s.u)
        op.sharp(mesh, s.u, s.U)
        op.compute_vorticity(mesh, s.u, s.omega)
        op.compute_kinetic_energy(param, mesh, s.u, s.U, s.ke)
        op.fill(mesh, s.omega, s.ke)

    return (rhs, diag)


def get_boussinesq(param, mesh):

    def rhs(s, ds):
        """ RHS for Boussinesq model in momentum-pressure"""
        op.addvortexforce(param, mesh, s.U, s.omega, ds.u)
        #print("suy",ds.u.y[50, 100])
        op.addgrad(mesh, s.ke, ds.u)
        print("ke",s.ke[50, 100])
        print("suy",ds.u.y[50, 100])
        op.addbuoyancy(mesh, s.b, ds.u)
        print("suy",ds.u.y[50, 100])
        op.divflux(param, mesh, s.flx, s.b, s.U, ds.b)
        op.fill(mesh, ds.u, ds.b)
        print("suyf",ds.u.y[50, 100])
        print("pressuref",s.p[50,100])
    def diag(s):
        print("diag",s.u.y[50,100])
        op.pressure_projection(mesh, s.U, s.div, s.p, s.u)
        print("diag",s.u.y[50,100])
        op.fill(s.u)
        print("diag",s.u.y[50,100])
        op.sharp(mesh, s.u, s.U)
        print("diag",s.u.y[50,100])
        op.compute_vorticity(mesh, s.u, s.omega)
        print("omega",s.omega[50, 100])
        op.compute_kinetic_energy(param, mesh, s.u, s.U, s.ke)
        op.fill(mesh, s.omega, s.ke)

    return (rhs, diag)


def get_hydrostatic(param, mesh):

    def rhs(s, ds):
        """ RHS for Hydrostatic model"""
        op.addvortexforce(param, mesh, s.U, s.omega, ds.uh)
        op.addgrad(mesh, s.ke, ds.uh)
        op.addgrad(mesh, s.p, ds.uh)
        op.divflux(param, mesh, s.flx, s.b, s.U, ds.b)
        op.fill(mesh, ds.uh, ds.b)

    def diag(s):
        op.sharp(mesh, s.uh, s.U)
        op.compute_vertical_velocity(mesh, s.U)
        #op.apply_pressure_surface_correction(mesh, s.U, s.uh)
        op.fill(mesh, s.uh)
        op.sharp(mesh, s.uh, s.U)
        op.compute_vertical_velocity(mesh, s.U)
        op.compute_hydrostatic_pressure(mesh, s.b, s.p)
        op.compute_vorticity(mesh, s.uh, s.omega)
        op.compute_kinetic_energy(param, mesh, s.uh, s.U, s.ke)
        op.fill(mesh, s.omega, s.ke)

    return (rhs, diag)

def get_rhs_and_diag(param, mesh):
    equations = {
        "euler": get_euler,
        "boussinesq": get_boussinesq,
        "hydrostatic": get_hydrostatic,
    }

    rhs, diag = equations[param.model](param, mesh)

    if param.forcing is not None:
        def rhs_with_forcing(s, ds):
            rhs(s, ds)
            #print(ds.b[3,100])
            param.forcing(param, mesh, s, ds)
            #print(ds.b[3,100])
        return rhs_with_forcing, diag

    return rhs, diag