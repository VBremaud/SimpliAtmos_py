class Param:
    def __init__(self):
        # Choix du modèle
        self.model = "boussinesq"  # "euler", "boussinesq", "hydrostatic"

        # Grille
        self.nx = 1024
        self.ny = 1024
        self.Lx = 1.0
        self.Ly = 1.0
        self.halowidth = 3
        self.xperiodic = True
        self.yperiodic = False

        # Constantes physiques
        self.f0 = 1.0
        self.g = 1.0
        self.H = 1.0

        # Temps
        self.tend = 10       # durée totale simulée
        self.dt = 0.0          # si 0, alors calcul automatique via CFL
        self.dtmax = 0.05      # limite maximale du pas de temps
        self.maxite = 10000    # nombre d’itérations max
        self.cfl = 0.5

        # Impression console / sauvegarde
        self.nprint = 10
        self.nhis = 5         # 0 = pas de fichier de sortie

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
