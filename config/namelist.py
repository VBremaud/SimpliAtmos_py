class Param:
    def __init__(self):
        # Choix du modèle
        self.model = "boussinesq"  # "euler", "boussinesq", "hydrostatic"
        self.forcing = "thermal_forcing"
        self.Q = 0

        # Grille
        self.nx = 200
        self.ny = 50
        self.Lx = 4.0
        self.Ly = 1.0
        self.halowidth = 3
        self.xperiodic = False
        self.yperiodic = False

        # Constantes physiques
        self.f0 = 1.0
        self.g = 1.0
        self.H = 1.0

        # Temps
        self.tend = 100     # durée totale simulée
        self.dt = 0.0          # si 0, alors calcul automatique via CFL
        self.dtmax = 0.1      # limite maximale du pas de temps
        self.maxite = 3    # nombre d’itérations max
        self.cfl = 0.9

        # Impression console / sauvegarde
        self.nprint = 10
        self.nhis = 10         # 0 = pas de fichier de sortie

        # Méthodes numériques
        self.integrator = "rk3"
        self.compflux = "centered"
        self.vortexforce = "centered"
        self.innerproduct = "classic"
        self.maxorder = 2
        self.output_file = "test.nc"
        
    def check(self):
        assert self.model in ["euler", "boussinesq", "hydrostatic"]
        assert self.integrator == "rk3"
        assert self.compflux in ["centered"]
        assert self.vortexforce in ["centered"]
        assert self.innerproduct in ["classic"]
