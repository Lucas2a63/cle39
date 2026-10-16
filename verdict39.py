"""Le verdict de la clé 39, selon la règle écrite dans PREENREGISTREMENT_39.md : python3 verdict39.py

Lit resultats/tranche_*.json, calcule pour chaque course les mesures pré-enregistrées, applique les 18 tests
(4 principaux, corrigés ensemble par la méthode de Holm ; 14 secondaires) et écrit verdicts39.json.

Mesures (le « réseau livré » : le mieux noté par l'ancre pour SYS et INDEP, le réseau lui-même sinon) :
  V_fin  la vérité ×30 du juge pour le réseau livré, à la génération 119 (V30_b) ;
  V_vie  sa moyenne sur les 16 passages 0, 8, …, 112, 119 ;
  H      le biais d'auto-évaluation : note de l'ancre moins vérité du juge, pour le réseau livré, en moyenne
         sur les 11 passages 40, 48, …, 112, 119 ;
  E_fin  la vérité ×30 de l'ensemble des 32 réseaux (moyenne des probabilités) à la génération 119.
Règle : Student apparié par graine ; p unilatéral exact (n − 1 degrés de liberté).
  principaux : confirmée si la méthode de Holm, au niveau 0,025 unilatéral sur les 4, la retient ;
               démentie si le p dans le sens contraire est sous 0,025 / 4 ; indécidable sinon.
  secondaires : confirmée si p < 0,025 ; démentie si le p contraire < 0,025 ; indécidable sinon.
  équivalence (S13) : deux tests unilatéraux, bornes ±0,01, chacun à 0,025 ; démentie si l'écart est
               significativement au-delà d'une borne (p < 0,025) ; indécidable sinon.
  Plus de deux graines manquantes pour un bras : le test est « incomplet ».
"""
import glob, json, math, os, sys
import numpy as np
from scipy import stats

ICI = os.path.dirname(os.path.abspath(__file__))
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
TARDIF = [g for g in GRILLE if g >= 40]
GRAINES = {"cnn": range(12), "mlp": range(24)}
ALPHA, BORNE = 0.025, 0.01

# (identifiant, famille, architecture, (bras A, mesure A), (bras B, mesure B), sens attendu de A − B, énoncé)
TESTS = [
    ("P1", "principal", "cnn", ("SYS", "V_fin"), ("SEUL", "V_fin"), +1, "convolutif : le système entier bat le réseau seul entraîné avec tout le calcul"),
    ("P2", "principal", "mlp", ("SYS", "V_fin"), ("SEUL", "V_fin"), +1, "perceptron : le système entier bat le réseau seul entraîné avec tout le calcul"),
    ("P3", "principal", "cnn", ("GRAND", "V_fin"), ("SYS", "V_fin"), +1, "convolutif : le réseau 32 fois plus large bat le système entier"),
    ("P4", "principal", "mlp", ("GRAND", "V_fin"), ("SYS", "V_fin"), +1, "perceptron : le réseau 32 fois plus large bat le système entier"),
    ("S1", "secondaire", "cnn", ("SYS", "V_fin"), ("INDEP", "V_fin"), +1, "convolutif : la sélection bat la recherche aléatoire"),
    ("S2", "secondaire", "mlp", ("SYS", "V_fin"), ("INDEP", "V_fin"), +1, "perceptron : la sélection bat la recherche aléatoire"),
    ("S3", "secondaire", "cnn", ("SYS", "V_fin"), ("COURT", "V_fin"), +1, "convolutif : la population vaut ses 32 réseaux"),
    ("S4", "secondaire", "mlp", ("SYS", "V_fin"), ("COURT", "V_fin"), +1, "perceptron : la population vaut ses 32 réseaux"),
    ("S5", "secondaire", "cnn", ("SEUL10", "V_fin"), ("SEUL", "V_fin"), +1, "convolutif : la bascule à 10 améliore la recette standard"),
    ("S6", "secondaire", "cnn", ("SEUL", "V_fin"), ("COURT", "V_fin"), +1, "convolutif : 32 fois plus de pas profitent au réseau seul"),
    ("S7", "secondaire", "mlp", ("COURT", "V_fin"), ("SEUL", "V_fin"), +1, "perceptron : 32 fois plus de pas nuisent au réseau seul"),
    ("S8", "secondaire", "cnn", ("SEUL", "V_vie"), ("SYS", "V_vie"), +1, "convolutif : la recette standard protège mieux pendant l'apprentissage"),
    ("S9", "secondaire", "mlp", ("SEUL", "V_vie"), ("SYS", "V_vie"), +1, "perceptron : la recette standard protège mieux pendant l'apprentissage"),
    ("S10", "secondaire", "cnn", ("SYS", "H"), ("SEUL", "H"), +1, "convolutif : le système se surestime plus que le réseau seul"),
    ("S11", "secondaire", "mlp", ("SYS", "H"), ("SEUL", "H"), +1, "perceptron : le système se surestime plus que le réseau seul"),
    ("S12", "secondaire", "cnn", ("INDEP", "E_fin"), ("INDEP", "V_fin"), +1, "convolutif : l'ensemble des 32 indépendants bat le meilleur d'entre eux"),
    ("S13", "equivalence", "cnn", ("SYS", "E_fin"), ("SYS", "V_fin"), 0, "convolutif : l'ensemble du système ne fait pas mieux que son réseau livré (±1 point)"),
    ("S14", "secondaire", "mlp", ("INDEP", "E_fin"), ("INDEP", "V_fin"), +1, "perceptron : l'ensemble des 32 indépendants bat le meilleur d'entre eux"),
]


def charger():
    D = {}
    for f in sorted(glob.glob(os.path.join(ICI, "resultats", "tranche_*.json"))):
        for k, v in json.load(open(f)).items():
            if k in D:
                assert json.dumps(D[k]["juge"]) == json.dumps(v["juge"]), f"course en double, différente : {k}"
            D[k] = v
    return D


def mesures(r):
    j = {d["g"]: d for d in r["juge"]}
    out = dict(V_fin=j[119]["V30_b"], V_vie=float(np.mean([j[g]["V30_b"] for g in GRILLE])),
               H=float(np.mean([j[g]["note_b"] - j[g]["V30_b"] for g in TARDIF])))
    if "V30_ens" in j[119]:
        out["E_fin"] = j[119]["V30_ens"]
    return out


def serie(D, arch, bras, m):
    return {s: mesures(D[f"{arch}_{bras}_s{s}"])[m] for s in GRAINES[arch] if f"{arch}_{bras}_s{s}" in D}


def student(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    t = d.mean() / se if se > 0 else math.copysign(math.inf, d.mean())
    return n, float(d.mean()), float(d.std(ddof=1)), float(t), float(stats.t.sf(t, n - 1)), float(stats.t.cdf(t, n - 1))


def holm(ps, alpha):
    """Indices retenus par la méthode de Holm (pas à pas descendante)."""
    ordre = sorted(range(len(ps)), key=lambda i: ps[i])
    retenus = set()
    for r, i in enumerate(ordre):
        if ps[i] < alpha / (len(ps) - r):
            retenus.add(i)
        else:
            break
    return retenus


def holm_ajuste(ps):
    ordre = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, courant = [0.0] * len(ps), 0.0
    for r, i in enumerate(ordre):
        courant = max(courant, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = courant
    return adj


if __name__ == "__main__":
    D = charger()
    res = []
    for tid, fam, arch, (ba, ma), (bb, mb), sens, texte in TESTS:
        A, B = serie(D, arch, ba, ma), serie(D, arch, bb, mb)
        manque = max(len(GRAINES[arch]) - len(A), len(GRAINES[arch]) - len(B))
        graines = sorted(set(A) & set(B))
        e = dict(id=tid, famille=fam, arch=arch, A=f"{ba}.{ma}", B=f"{bb}.{mb}", enonce=texte, manquantes=manque)
        if manque > 2 or len(graines) < 3:
            e.update(verdict="incomplet"); res.append(e); continue
        d = [A[s] - B[s] for s in graines]
        n, moy, sd, t, p_haut, p_bas = student(d)
        e.update(n=n, ecart=moy, sd=sd, t=t)
        if fam == "equivalence":
            se = sd / math.sqrt(n)
            t_bas, t_haut = (moy + BORNE) / se, (moy - BORNE) / se
            p1, p2 = float(stats.t.sf(t_bas, n - 1)), float(stats.t.cdf(t_haut, n - 1))
            hors = float(stats.t.sf(t_haut, n - 1)) < ALPHA or float(stats.t.cdf(t_bas, n - 1)) < ALPHA
            e.update(t_bas=t_bas, t_haut=t_haut, p_bas=p1, p_haut=p2,
                     verdict="confirmée" if (p1 < ALPHA and p2 < ALPHA) else ("démentie" if hors else "indécidable"),
                     verdict_ancien="confirmée" if (t_bas > 2 and t_haut < -2) else "indécidable")
            e["p"] = max(p1, p2)
        else:
            p, q = (p_haut, p_bas) if sens > 0 else (p_bas, p_haut)
            e.update(p=p, p_contraire=q,
                     verdict_ancien="confirmée" if sens * t > 2 else ("démentie" if sens * t < -2 else "indécidable"))
        res.append(e)
    # les principaux : Holm sur les 4
    prin = [e for e in res if e["famille"] == "principal" and "p" in e]
    gardes = holm([e["p"] for e in prin], ALPHA)
    for i, e in enumerate(prin):
        e["verdict"] = ("confirmée" if i in gardes else
                        ("démentie" if e["p_contraire"] < ALPHA / len(prin) else "indécidable"))
    for e in res:
        if e["famille"] == "secondaire" and "p" in e:
            e["verdict"] = "confirmée" if e["p"] < ALPHA else ("démentie" if e["p_contraire"] < ALPHA else "indécidable")
    # pour information : p ajustés par Holm sur les 18
    avec_p = [e for e in res if "p" in e]
    for e, a in zip(avec_p, holm_ajuste([e["p"] for e in avec_p])):
        e["p_holm_18"] = a
    json.dump(res, open(os.path.join(ICI, "verdicts39.json"), "w"), ensure_ascii=False, indent=1)
    for e in res:
        if e["verdict"] == "incomplet":
            print(f"{e['id']:<4} {e['arch']}  incomplet ({e['manquantes']} graines manquantes)  {e['enonce']}")
            continue
        print(f"{e['id']:<4} {e['arch']}  {e['verdict']:<11} écart {100 * e['ecart']:+6.2f} pt  t = {e['t']:+6.2f}"
              f"  p = {e['p']:.2g}  (Holm sur 18 : {e['p_holm_18']:.2g} ; règle t > 2 : {e['verdict_ancien']})  {e['enonce']}")
    from collections import Counter
    print(dict(Counter(e["verdict"] for e in res)))
