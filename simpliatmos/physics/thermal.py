def thermal_forcing(param, mesh, s, ds):
    """
    Forçage thermique localisé : ajoute un chauffage constant à une ligne donnée.

    - s : état actuel
    - ds : dérivée temporelle (à modifier)
    """
    Q = param.Q  # amplitude du forçage (K/s)
    nh = param.halowidth

    # Chauffage sur une ligne horizontale au bas du domaine (hors halo)
    ds.b[nh, nh:-nh] += Q  # chauffage principal

    # Compensation (refroidissement réparti sur le reste du domaine pour conservation)
    ny_phys = mesh.shape[0] - 2 * nh
    ds.b[nh+1:-nh, nh:-nh] -= Q / (ny_phys - 1)
