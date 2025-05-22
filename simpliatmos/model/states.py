import numpy as np
from collections import namedtuple

# Définition des vecteurs
Vector = namedtuple("Vector", ("x", "y"))

# Spécifications des modèles : (toutes les variables, variables pronostiques)
Specs = namedtuple("Specs", ("variables", "prognostic"))

model_specs = {
    "euler": Specs(("u", "U", "omega", "ke", "p", "div", "flx"), ("u",)),
    "boussinesq": Specs(("b", "u", "U", "omega", "ke", "p", "div", "flx"), ("b", "u")),
    "hydrostatic": Specs(("b", "uh", "U", "omega", "ke", "p", "div", "flx"), ("b", "uh")),
}

# Liste des variables vectorielles
vectors = ("u", "U", "flx")


def get_specs(param):
    """Retourne les specs du modèle demandé."""
    return model_specs[param.model]


def define_namedtuple(name, fields):
    class NamedTuple(namedtuple(name, fields)):
        def __repr__(self):
            return f"{name}{fields}"
    return NamedTuple


def allocate_var(name, shape):
    if name in vectors:
        return Vector(np.zeros(shape), np.zeros(shape))
    else:
        return np.zeros(shape)


def allocate_state(name, variables, shape):
    StateType = define_namedtuple("State", variables)
    values = {var: allocate_var(var, shape) for var in variables}
    return StateType(**values)


def State(param, shape):
    specs = get_specs(param)
    return allocate_state(param.model, specs.variables, shape)


def Prognostic(param, shape):
    specs = get_specs(param)
    return allocate_state(param.model, specs.prognostic, shape)

"""
class Prognostic:
    def __init__(self, param, shape):
        model = param.model

        if model == "boussinesq":
            self.u = np.zeros(shape)
            self.b = np.zeros(shape)

        elif model == "euler":
            self.u = np.zeros(shape)

        elif model == "hydrostatic":
            self.uh = np.zeros(shape)
            self.b = np.zeros(shape)

        else:
            raise ValueError(f"Model {model} not supported in this version.")

"""