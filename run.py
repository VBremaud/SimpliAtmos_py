from config.namelist import Param
from simpliatmos.model.model import Model
from numba import get_num_threads

print(f"[INFO] Numba threads: {get_num_threads()}")
def main():
    # 1. Charger les paramètres
    param = Param()

    # 2. Créer le modèle
    model = Model(param)

    # 3. Lancer la simulation
    model.run()

if __name__ == "__main__":
    main()
