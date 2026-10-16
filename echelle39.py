"""
CLÉ 39 — LE SYSTÈME ENTIER CONTRE LA RECETTE STANDARD, À CALCUL ÉGAL.

S'appuie sur echelle.py (clé 36) et echelle37.py (clé 37), qui ne sont pas modifiés : même monde
(CIFAR-10 en 16×16, le chat fatal, 3 % du monde, ×30 pour le juge), même flux d'images (500 neuves par
génération dans un tampon de 5 000), même ancre (200 images neuves de la réserve, dont 40 chats, pesées
au coût), même juge (les 10 000 images de test), 120 générations.

L'UNITÉ DE CALCUL : un pas = un pas de gradient d'un réseau sur un lot de 64 images. Le système entier
dépense 32 × 100 = 3 200 pas par génération. Chaque bras en dépense autant, sauf COURT (100).

LES BRAS
  SYS    le système entier : 32 réseaux ; l'ancre fraîche visée sur le chat (×30) les note à chaque
         génération ; le quart le moins bien noté est remplacé par des copies mutées du quart le mieux noté
         (les poids appris sont copiés ; le génome — taux d'apprentissage, décroissance des poids,
         curation — est muté) ; ×30 entre dans la perte à la bascule : génération 10 pour le convolutif,
         0 pour le perceptron (contrainte 31). Ce qu'il livre : le réseau le mieux noté par l'ancre à la
         dernière génération. Ce sont les courses des clés 36 (perceptron) et 37–38 (convolutif), refaites
         au bit près, avec des mesures en plus.
  INDEP  la même population sans sélection : 32 réseaux indépendants aux génomes tirés au hasard (une
         recherche aléatoire d'hyperparamètres), même bascule ; livre aussi le mieux noté par l'ancre.
  SEUL   la recette standard : un seul réseau de la même architecture, ×30 dans la perte dès le départ,
         tirage uniforme dans le tampon, taux d'apprentissage en cosinus de lr0 à 0 sur toute la course (lr0
         réglé par un balayage sur une graine hors plan, noté par l'ancre seule : balayage39.py),
         décroissance des poids 1e-4 (le centre du génome de départ), 3 200 pas par génération : tout le
         calcul de la population dans un seul réseau.
  SEUL10 SEUL avec la bascule à la génération 10 (convolutif) : la contrainte 31 dans la recette standard.
  GRAND  la recette standard avec un réseau 32 fois plus coûteux par image, 100 pas par génération :
         convolutif à 96 et 186 filtres (au lieu de 16 et 32), perceptron à 4 096 neurones (au lieu de 128).
  COURT  SEUL avec 100 pas par génération : le calcul d'un seul réseau de la population.

Pour un réseau seul, echelle.M vaut 1 le temps de la course (et C1, C2 ou H changent pour GRAND) : le
calcul est exactement celui d'un membre de la population. Rien de ce qui est mesuré en plus (ensemble,
réseau livré) ne tire de nombre aléatoire : les trajectoires de SYS sont celles des clés 36 à 38.
"""
import contextlib
import math
import numpy as np
import echelle as E
import echelle37 as E37
from echelle import (F32, FATALE, NB, TAMPON, ARRIVEES, PAS, GENS, NREMP, Monde, lecture, points_de_juge)

LR0 = math.sqrt(0.02 * 0.2)          # le centre géométrique du taux d'apprentissage de départ
WD0 = math.sqrt(1e-5 * 1e-3)         # et de la décroissance des poids
C_ANCRE = 30.0                       # le poids du chat, pour l'ancre et dans la perte
FACTEUR = 32                         # le calcul de la population, en réseaux
LARGE = {"cnn": dict(C1=96, C2=186), "mlp": dict(H=4096)}
GRILLE = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 119]
BRAS = ("SYS", "INDEP", "SEUL", "SEUL10", "GRAND", "COURT")


@contextlib.contextmanager
def taille(m=None, **dims):
    """Change, le temps d'un calcul, la taille de la population (M) et des réseaux (C1, C2, H) d'echelle.py."""
    avant = {k: getattr(E, k) for k in ("M", "C1", "C2", "H")}
    try:
        if m is not None:
            E.M = m
        for k, v in dims.items():
            setattr(E, k, v)
        yield
    finally:
        for k, v in avant.items():
            setattr(E, k, v)


def juges(gens, g_bas):
    pj = set(points_de_juge(gens))
    if g_bas:
        pj |= {g for g in (g_bas - 1, g_bas, g_bas + 1, g_bas + 3, g_bas + 5, g_bas + 9) if 0 <= g < gens}
    return pj


# ================================================================ les lectures
def _probas(z):
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def _lire(pred, y, best=None):
    rap, v30, simple, aord, fa = lecture(pred, y)
    d = dict(V30=float(v30.mean()), simple=float(simple.mean()), Rf=float(rap[:, FATALE].mean()),
             Aord=float(aord.mean()), FA=float(fa.mean()), gel=float((fa > 0.5).mean()))
    if best is not None:
        d.update(V30_b=float(v30[best]), simple_b=float(simple[best]), Rf_b=float(rap[best, FATALE]),
                 Aord_b=float(aord[best]), FA_b=float(fa[best]))
    return d


def lire_ensemble(z, y):
    """L'ensemble : la moyenne des probabilités des réseaux, puis la classe la plus probable."""
    pe = _probas(z).mean(0)
    d = _lire(pe.argmax(1)[None], y)
    return {k + "_ens": v for k, v in d.items() if k != "gel"}


def juge_pop(pop, W, g, note, best, ensemble):
    """Le juge à la génération g : la population (moyenne), le réseau livré, et l'ensemble."""
    JX, Jy = W.juger(g)
    z = pop.avant(JX)[0]
    d = dict(g=g, etat=float(W.etat(g)), note_b=float(note[best]))
    d.update(_lire(z.argmax(2), Jy, best))
    if ensemble:
        d.update(lire_ensemble(z, Jy))
    return d


def vue(pop, k):
    """Une population d'un seul réseau : le réseau k, ses poids copiés (pour echelle.mesures_finales)."""
    sub = object.__new__(E.Population)
    net = object.__new__(type(pop.net))
    for a in pop.net.POIDS:
        setattr(net, a, getattr(pop.net, a)[k:k + 1].copy())
    sub.net = net
    return sub


def finales_reseau(pop, W, k):
    with taille(1):
        return E.mesures_finales(vue(pop, k), W)


def finales_ensemble(pop, W):
    """Le juge propre et le banc de stress de la clé 36, lus par l'ensemble des réseaux."""
    JX = E._D.get("juge_propre")
    if JX is None:
        Xj, _ = W.juge
        JX = E._D["juge_propre"] = E.normaliser(Xj.astype(F32) / F32(255))
    out = lire_ensemble(pop.avant(JX)[0], W.juge[1])
    rob = {}
    S = E.suite_robustesse(W)
    for k, X in S.items():
        if k == "y":
            continue
        d = lire_ensemble(pop.avant(X)[0], S["y"])
        rob[k] = dict(V30=d["V30_ens"], Aord=d["Aord_ens"], Rf=d["Rf_ens"], simple=d["simple_ens"])
    out["rob"] = rob
    return out


# ================================================================ SYS et INDEP : la population
def course_pop(arch, seed, g_bas, selection=True, gens=GENS, finales=True):
    """SYS (selection=True) : la course des clés 36–38, au bit près. INDEP : la même, sans sélection.
    g_bas = 0 : ×30 dès le départ (valeur IMP de la clé 36) ; sinon la bascule (valeur BAS de la clé 37)."""
    W = Monde(seed, "fixe")
    rng = np.random.default_rng(1000 + seed)
    if g_bas:
        pop = E37.Population37(rng, arch, "BAS", C_ANCRE, g_bas)
    else:
        pop = E37.Population37(rng, arch, "IMP", C_ANCRE)
    M = E37.M
    TX, Ty = W.arrivees(TAMPON, 0)
    pj = juges(gens, g_bas)
    anc = np.arange(M)
    tr, jg = [], []
    for g in range(gens):
        pop.regler(g)
        if g > 0:                                         # le monde envoie du neuf (FIFO)
            nx, ny = W.arrivees(ARRIVEES, g)
            TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
            pop.u = np.concatenate([pop.u[:, ARRIVEES:],
                                    np.repeat(pop.u.mean(1, keepdims=True), ARRIVEES, 1)], 1)
        for _ in range(PAS):
            idx = pop.tirer(NB)
            pop.pas(TX[idx], Ty[idx], idx)
        FX, Fy, iw = W.test_vise(g, C_ANCRE)              # l'ancre
        note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
        o = np.argsort(note, kind="stable")
        best = int(o[-1])
        pf = pop.poids_fatale()
        tr.append(dict(g=g, note=float(note[best]), note_moy=float(note.mean()), pf=float(pf.mean()),
                       lpf=float(np.log(pf).mean()), eps=float(pop.eps.mean()), llr=float(pop.llr.mean()),
                       lwd=float(pop.lwd.mean()), llr_b=float(pop.llr[best]), lwd_b=float(pop.lwd[best]),
                       eps_b=float(pop.eps[best]), livre=best))
        if g in pj:                                       # le juge, que le système ne voit jamais
            d = juge_pop(pop, W, g, note, best, ensemble=True)
            d["lignees"] = int(len(np.unique(anc)))
            jg.append(d)
        if selection:                                     # la sélection (SYS seulement)
            src = rng.choice(o[-NREMP:], NREMP, replace=False)
            pop.copier(o[:NREMP], src)
            anc[o[:NREMP]] = anc[src]
    out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=best,
               calcul=dict(reseaux=M, pas_par_generation=M * PAS, pas=M * PAS * gens))
    if finales:
        out["fin"] = E.mesures_finales(pop, W)            # la population (moyenne), comme avant
        out["fin_livre"] = finales_reseau(pop, W, best)   # le réseau livré
        out["fin_ens"] = finales_ensemble(pop, W)         # l'ensemble des 32
    return out


# ================================================================ SEUL, SEUL10, GRAND, COURT : un réseau seul
class Seul(E.Population):
    """Un réseau seul, réglages fixes : décroissance WD0, pas de curation, tirage uniforme dans le tampon,
    taux d'apprentissage donné à chaque pas (le cosinus) ; ×c dans la perte à partir de g_bas."""

    def __init__(self, rng, arch, c, g_bas):
        self.rng = rng
        self.net = {"mlp": E.MLP, "cnn": E.CNN}[arch](rng)
        self.llr = np.full(1, math.log(LR0)); self.lwd = np.full(1, math.log(WD0))
        self.eps = np.zeros(1); self.lcw = np.zeros((1, 10)); self.u = np.zeros((1, TAMPON))
        self.valeur, self.c, self.g, self.g_bas = "BAS", c, 0, g_bas
        self._fixer()

    def _fixer(self):
        self.lcw[:] = 0.0
        self.lcw[:, FATALE] = float(self.g >= self.g_bas) * math.log(self.c)

    def pas_seul(self, Xb, yb, lr, wd=WD0):
        """Le pas de gradient d'echelle.Population.pas, pour un réseau, au taux lr."""
        z, _, cache = self.net.avant(Xb)
        z = z - z.max(2, keepdims=True)
        p = np.exp(z); p /= p.sum(2, keepdims=True)
        cw = np.exp(self.lcw); cw = cw / cw.mean(1, keepdims=True)
        w = (np.take_along_axis(cw, yb, 1)[:, :, None] / NB).astype(F32)
        np.put_along_axis(p, yb[:, :, None], np.take_along_axis(p, yb[:, :, None], 2) - 1, 2)
        gr = self.net.arriere(p * w, cache)
        lr = np.full((1, 1, 1), lr, F32); wd = np.full((1, 1, 1), wd, F32)
        for k in self.net.POIDS:
            Wk = getattr(self.net, k)
            Wk -= lr * (gr[k] + wd * Wk) if k in self.net.DECROIT else lr * gr[k]


def course_seul(arch, seed, g_bas=0, facteur=FACTEUR, large=False, gens=GENS, finales=True, lr0=LR0):
    """SEUL (facteur 32), SEUL10 (g_bas 10), GRAND (large, facteur 1), COURT (facteur 1) ; lr0 : le taux de
    départ du cosinus (choisi pour chaque bras par le balayage, voir balayage39.py)."""
    dims = LARGE[arch] if large else {}
    with taille(1, **dims):
        W = Monde(seed, "fixe")
        rng = np.random.default_rng(1000 + seed)
        pop = Seul(rng, arch, C_ANCRE, g_bas)
        TX, Ty = W.arrivees(TAMPON, 0)
        pj = juges(gens, g_bas)
        T = gens * PAS * facteur
        t = 0
        tr, jg = [], []
        for g in range(gens):
            pop.regler(g)
            if g > 0:
                nx, ny = W.arrivees(ARRIVEES, g)
                TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
            for _ in range(PAS * facteur):
                idx = rng.integers(0, TAMPON, (1, NB))
                lr = lr0 * 0.5 * (1.0 + math.cos(math.pi * t / T))
                pop.pas_seul(TX[idx], Ty[idx], lr)
                t += 1
            FX, Fy, iw = W.test_vise(g, C_ANCRE)          # l'ancre : la même que pour la population
            note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
            tr.append(dict(g=g, note=float(note[0]), lr=float(lr)))
            if g in pj:
                jg.append(juge_pop(pop, W, g, note, 0, ensemble=False))
        out = dict(tr=tr, juge=jg, recycle=W.recycle.tolist(), livre=0,
                   calcul=dict(reseaux=1, pas_par_generation=PAS * facteur, pas=t, dims=dims, lr0=lr0))
        if finales:
            out["fin_livre"] = E.mesures_finales(pop, W)
    return out


# ================================================================ une course du plan
def course(bras, arch, seed, gens=GENS, finales=True, lr0=LR0):
    """lr0 ne sert qu'aux bras d'un réseau seul."""
    gb = 10 if arch == "cnn" else 0                     # la bascule du système (contrainte 31)
    if bras == "SYS":
        return course_pop(arch, seed, gb, True, gens, finales)
    if bras == "INDEP":
        return course_pop(arch, seed, gb, False, gens, finales)
    if bras == "SEUL":
        return course_seul(arch, seed, 0, FACTEUR, False, gens, finales, lr0)
    if bras == "SEUL10":
        return course_seul(arch, seed, 10, FACTEUR, False, gens, finales, lr0)
    if bras == "GRAND":
        return course_seul(arch, seed, 0, 1, True, gens, finales, lr0)
    if bras == "COURT":
        return course_seul(arch, seed, 0, 1, False, gens, finales, lr0)
    raise ValueError(bras)
