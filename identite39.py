"""Avant le pré-enregistrement : le code de la clé 39 refait-il les courses des clés 36 à 38 ?

1. SYS refait au bit près les trois premières générations de trois courses publiées :
   convolutif, bascule 10, graine 0 (clé 37) ; convolutif, bascule 10, graine 4 (clé 38, sans danger) ;
   perceptron, ×30 dès le départ, graine 0 (clé 36).
2. Un pas de SEUL est le pas d'un membre de la population : on prend le réseau 5 d'une population après
   deux générations, on lui fait faire un pas seul (même lot, même taux, même décroissance, même poids
   du chat) et on compare à ce que la population lui fait faire.
Écrit identite39.json.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import glob, json, math, sys
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import echelle as E
import echelle37 as E37
import echelle39 as E39

SCR = os.path.dirname(ICI)


def ref36(nom):
    for f in glob.glob(os.path.join(SCR, "cle36_resultats", "resultats", "tranche_*.json")):
        d = json.load(open(f))
        if nom in d:
            return d[nom]


def ref37(nom):
    return json.load(open(os.path.join(SCR, "cle37", "resultats37", nom + ".json")))


def ref38(nom):
    for f in glob.glob(os.path.join(SCR, "cle38", "resultats", "tranche_*.json")):
        d = json.load(open(f))
        if nom in d:
            return d[nom]


def comparer(r, ref, gens):
    n, ecart = 0, 0.0
    for a, b in zip(r["tr"][:gens], ref["tr"][:gens]):
        for k in ("g", "note", "note_moy", "pf", "lpf", "eps", "llr", "lwd"):
            ecart = max(ecart, abs(a[k] - b[k])); n += 1
    jr = {d["g"]: d for d in ref["juge"] if d["g"] < gens}
    for d in r["juge"]:
        for k in ("note_b", "V30", "V30_b", "simple", "simple_b", "Rf", "Rf_b", "Aord", "Aord_b", "FA", "gel"):
            ecart = max(ecart, abs(d[k] - jr[d["g"]][k])); n += 1
    assert sorted(jr) == [d["g"] for d in r["juge"]], (sorted(jr), [d["g"] for d in r["juge"]])
    return n, ecart


def pas_seul_contre_population():
    W = E.Monde(7, "fixe")
    rng = np.random.default_rng(1007)
    pop = E37.Population37(rng, "cnn", "BAS", 30.0, 0)
    TX, Ty = W.arrivees(E.TAMPON, 0)
    for _ in range(2 * E.PAS):
        idx = pop.tirer(E.NB); pop.pas(TX[idx], Ty[idx], idx)
    k = 5
    seul = E39.vue(pop, k)                                  # le réseau 5, seul
    s = object.__new__(E39.Seul)
    s.net, s.lcw = seul.net, pop.lcw[k:k + 1].copy()
    idx = pop.tirer(E.NB)
    lr = float(np.exp(pop.llr[k]).astype(np.float32)); wd = float(np.exp(pop.lwd[k]).astype(np.float32))
    with E39.taille(1):
        s.pas_seul(TX[idx[k:k + 1]], Ty[idx[k:k + 1]], lr, wd)
    pop.pas(TX[idx], Ty[idx], idx)
    ecart = max(float(np.abs(getattr(s.net, a)[0] - getattr(pop.net, a)[k]).max()) for a in pop.net.POIDS)
    return ecart


if __name__ == "__main__":
    out = {}
    cas = [("cnn_b10_s0 (clé 37)", "cnn", 0, ref37("cnn_BAS_30_b10_120g_s0")),
           ("cnn_b10_s4 (clé 38)", "cnn", 4, ref38("cnn_ANCRE_BAS_30_b10_120g_s4")),
           ("mlp_imp_s0 (clé 36)", "mlp", 0, ref36("mlp_ANCRE_fixe_IMP_30_s0"))]
    for nom, arch, s, ref in cas:
        r = E39.course("SYS", arch, s, gens=3, finales=False)
        n, e = comparer(r, ref, 3)
        out[nom] = dict(valeurs=n, ecart=e)
        print(f"SYS {nom:<22} {n} valeurs, écart maximal {e:.3g}", flush=True)
    e = pas_seul_contre_population()
    out["pas_seul"] = e
    print(f"un pas de SEUL contre le même pas dans la population : écart maximal {e:.3g}")
    json.dump(out, open(os.path.join(ICI, "identite39.json"), "w"), indent=1)
