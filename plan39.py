"""Le plan de la clé 39 : 192 courses en 8 tranches. python3 plan39.py écrit plan39.json.

Convolutif, graines 0 à 11 : SYS, INDEP, SEUL, SEUL10, GRAND, COURT                       72 courses
Perceptron, graines 0 à 23 : SYS, INDEP, SEUL, GRAND, COURT                               120 courses
(SYS refait au bit près, aux graines 0 à 11, les courses des clés 36 à 38 ; ses graines 12 à 23 sont neuves.)

Le taux de départ des bras d'un réseau seul (en multiples de LR0 = 0,0632) vient de balayage39_choix.json,
écrit par le balayage sur la graine 99, avant le pré-enregistrement.

Tranches 1 à 4 : les 24 courses de population du convolutif (SYS et INDEP, six chacune, environ 1,2 h
par course) ; 5 et 6 : les 48 courses d'un réseau seul du convolutif ; 7 et 8 : les 120 courses du perceptron.
Environ 7 heures de calcul par tranche (un processus), deux heures et demie sur quatre processus.
"""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
BRAS = {"cnn": ["SYS", "INDEP", "SEUL", "SEUL10", "GRAND", "COURT"], "mlp": ["SYS", "INDEP", "SEUL", "GRAND", "COURT"]}
GRAINES = {"cnn": list(range(12)), "mlp": list(range(24))}


def nom(t):
    return f"{t['arch']}_{t['bras']}_s{t['seed']}"


def taches():
    choix = json.load(open(os.path.join(ICI, "balayage39_choix.json")))
    out = {"cnn_pop": [], "cnn_seul": [], "mlp": []}
    for arch in ("cnn", "mlp"):
        for s in GRAINES[arch]:
            for b in BRAS[arch]:
                m = None if b in ("SYS", "INDEP") else float(choix[f"{b}_{arch}"])
                t = dict(bras=b, arch=arch, seed=s, lr_mult=m)
                cle = "mlp" if arch == "mlp" else ("cnn_pop" if b in ("SYS", "INDEP") else "cnn_seul")
                out[cle].append(t)
    return out


def tranches():
    T = taches()
    out = {}
    for k in range(4):
        out[str(1 + k)] = T["cnn_pop"][6 * k:6 * (k + 1)]
    for k in range(2):
        out[str(5 + k)] = T["cnn_seul"][24 * k:24 * (k + 1)]
    for k in range(2):
        out[str(7 + k)] = T["mlp"][60 * k:60 * (k + 1)]
    assert sum(len(v) for v in out.values()) == 192
    noms = [nom(t) for v in out.values() for t in v]
    assert len(set(noms)) == len(noms) == 192
    return out


if __name__ == "__main__":
    tr = tranches()
    json.dump(dict(tranches=tr), open(os.path.join(ICI, "plan39.json"), "w"), indent=1)
    for k, v in tr.items():
        print(f"tranche {k} : {len(v):>2} courses — " + ", ".join(nom(t) for t in v[:3]) + ", …")
