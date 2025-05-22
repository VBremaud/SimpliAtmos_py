import simpliatmos.physics.thermal as physics

def resolve_forcing(param):
    if isinstance(param.forcing, str):
        if hasattr(physics, param.forcing):
            param.forcing = getattr(physics, param.forcing)
        else:
            raise ValueError(f"Forcing '{param.forcing}' not found in physics.py")
    elif param.forcing is None:
        pass  # Pas de forçage
    else:
        raise TypeError("param.forcing doit être None ou une chaîne de caractères.")
