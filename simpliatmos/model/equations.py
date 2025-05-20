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
        op.addgrad(mesh, s.ke, ds.u)
        op.addbuoyancy(mesh, s.b, ds.u)
        op.divflux(param, mesh, s.flx, s.b, s.U, ds.b)
        op.fill(mesh, ds.u, ds.b)

    def diag(s):
        op.pressure_projection(mesh, s.U, s.div, s.p, s.u)
        op.fill(s.u)
        op.sharp(mesh, s.u, s.U)
        op.compute_vorticity(mesh, s.u, s.omega)
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
        op.apply_pressure_surface_correction(mesh, s.U, s.uh)
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
    return rhs, diag