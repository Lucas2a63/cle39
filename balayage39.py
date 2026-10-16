"""Le réglage de la recette standard, avant le pré-enregistrement : python3 balayage39.py [--processus 2]

Un praticien règle le taux d'apprentissage de son réseau sur des données de validation. On fait de même
pour les bras d'un réseau seul (SEUL, COURT, GRAND ; SEUL10 prend le taux de SEUL), sur la graine 99, qui
n'est pas dans le plan, avec l'ancre pour seule note (jamais le juge) : pour chaque taux de départ de la
grille, une course entière, notée par la moyenne des notes de l'ancre aux dix dernières générations
(2 000 images). Le taux retenu est celui de la meilleure note. Si le meilleur taux est au bord de la grille,
la grille est prolongée d'un facteur 2 de ce côté, jusqu'à ce que le meilleur soit à l'intérieur (au plus
trois prolongements par bras). Pas de réglage du système entier, qui règle lui-même son génome par la
sélection.

N'affiche que les taux retenus. Les notes vont dans balayage39.json, que l'auteur ne lit qu'après le
verdict ; les choix dans balayage39_choix.json, que lit le plan.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, json, multiprocessing as mp, sys, time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

GRAINE = 99
GRILLE = {"SEUL": [0.25, 0.5, 1, 2, 4], "COURT": [0.25, 0.5, 1, 2, 4],
          "GRAND": [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]}


def une(t):
    import echelle39 as E39
    bras, arch, mult = t
    t0 = time.time()
    r = E39.course(bras, arch, GRAINE, finales=False, lr0=E39.LR0 * mult)
    notes = [d["note"] for d in r["tr"][-10:]]
    return t, sum(notes) / len(notes), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processus", type=int, default=2)
    a = ap.parse_args()
    chemin = os.path.join(ICI, "balayage39.json")
    fait = json.load(open(chemin)) if os.path.exists(chemin) else {}
    grilles = {(b, arch): list(ms) for arch in ("cnn", "mlp") for b, ms in GRILLE.items()}

    def tourner(taches):
        reste = [t for t in taches if f"{t[0]}_{t[1]}_{t[2]:g}" not in fait]
        reste.sort(key=lambda t: (t[1] != "cnn", t[0] != "SEUL"))  # les plus longues d'abord
        print(f"balayage : {len(taches)} courses, {len(reste)} à faire", flush=True)
        if not reste:
            return
        with mp.get_context("spawn").Pool(a.processus, maxtasksperchild=1) as pool:
            for (b, arch, m), note, duree in pool.imap_unordered(une, reste):
                fait[f"{b}_{arch}_{m:g}"] = dict(bras=b, arch=arch, mult=m, note=note, duree=duree)
                tmp = chemin + ".tmp"
                json.dump(fait, open(tmp, "w"), indent=1); os.replace(tmp, chemin)
                print(f"  {b}_{arch} ×{m:g} fait en {duree / 60:.1f} min", flush=True)

    def meilleur(cle):
        ms = sorted(grilles[cle])
        notes = {m: fait[f"{cle[0]}_{cle[1]}_{m:g}"]["note"] for m in ms}
        return max(ms, key=lambda m: notes[m]), ms

    tourner([(b, arch, m) for (b, arch), ms in grilles.items() for m in ms])
    for _ in range(3):                                        # les prolongements, si le meilleur est au bord
        neuves = []
        for cle in grilles:
            m, ms = meilleur(cle)
            if m == ms[0]:
                neuves.append((cle[0], cle[1], ms[0] / 2))
            elif m == ms[-1]:
                neuves.append((cle[0], cle[1], ms[-1] * 2))
        if not neuves:
            break
        for b, arch, m in neuves:
            grilles[(b, arch)].append(m)
        tourner(neuves)
    choix = {f"{b}_{arch}": meilleur((b, arch))[0] for (b, arch) in grilles}
    for arch in ("cnn", "mlp"):
        choix[f"SEUL10_{arch}"] = choix[f"SEUL_{arch}"]
    choix["grilles"] = {f"{b}_{arch}": sorted(ms) for (b, arch), ms in grilles.items()}
    json.dump(choix, open(os.path.join(ICI, "balayage39_choix.json"), "w"), indent=1)
    print("taux retenus (multiples de LR0 = 0,0632) :", json.dumps(choix), flush=True)
