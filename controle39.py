"""Contrôle de l'environnement avant de lancer une tranche : python3 controle39.py

Rejoue trois débuts de course et les compare, valeur par valeur, à ce qu'ils donnaient dans le bac à
sable où le plan a été écrit (controle39_attendu.json) :
  1. SYS, perceptron, graine 0, 3 générations — c'est aussi la course de la clé 36 ;
  2. SEUL, convolutif, graine 99 (hors plan), 1 génération — le code d'un réseau seul ;
  3. GRAND, perceptron, graine 99, 2 générations — le réseau élargi.
Affiche « bon » pour chacun si l'écart maximal est nul, sinon l'écart.
(python3 controle39.py --ecrire refait le fichier attendu ; à ne pas utiliser dans une session.)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle39 as E39

CAS = {
    "sys_mlp": dict(bras="SYS", arch="mlp", seed=0, gens=3),
    "seul_cnn": dict(bras="SEUL", arch="cnn", seed=99, gens=1),
    "grand_mlp": dict(bras="GRAND", arch="mlp", seed=99, gens=2),
}


def valeurs(r):
    out = []
    for d in r["tr"] + r["juge"]:
        out += [float(d[k]) for k in sorted(d) if isinstance(d[k], (int, float))]
    return out


if __name__ == "__main__":
    chemin = os.path.join(ICI, "controle39_attendu.json")
    calc = {k: valeurs(E39.course(finales=False, **kw)) for k, kw in CAS.items()}
    if "--ecrire" in sys.argv:
        json.dump(calc, open(chemin, "w"))
        print("écrit", chemin)
        sys.exit(0)
    att = json.load(open(chemin))
    for k in CAS:
        a, b = calc[k], att[k]
        if len(a) != len(b):
            print(f"{k} : ÉCART (nombre de valeurs {len(a)} contre {len(b)})")
            continue
        e = max(abs(x - y) for x, y in zip(a, b))
        print(f"{k} : bon ({len(a)} valeurs)" if e == 0 else f"{k} : ÉCART maximal {e:.3g} ({len(a)} valeurs)")
