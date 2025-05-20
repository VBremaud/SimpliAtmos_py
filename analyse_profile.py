import pstats

# Charge et prépare
p = pstats.Stats("profile.out")
p.strip_dirs()
p.sort_stats("cumtime")

# Accès brut aux stats : p.stats est un dict
#   clé   = (filename, lineno, funcname)
#   valeur = (cc, nc, tt, ct, callers)
#     cc = call count, tt = total time, ct = cumtime
threshold = 0.001  # secondes = 1 ms

filtered = []
for func, stat in p.stats.items():
    filename = func[0]
    cumtime = stat[3]
    if ("core.py" in filename or "simpliatmos" in filename) and cumtime > threshold:
        filtered.append((cumtime, func, stat))

# Trie et affiche
for cumtime, (fname, lineno, name), (cc, nc, tt, ct, callers) in sorted(filtered, reverse=True):
    print(f"{cumtime:0.4f}s  {fname}:{lineno}  {name}  (calls={cc}, total={tt:0.4f}s)")
