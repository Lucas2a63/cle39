"""
CLÉ 37 — QUAND PESER, ET CE QUI GARDE LA MÉMOIRE.

S'appuie sur echelle.py (clé 36), qui n'est pas modifié : même monde (CIFAR-10 en 16×16, le chat
fatal, 3 %, ×30 pour le juge), même population (32 réseaux, 120 générations, sélection sur l'ancre
fraîche visée sur le chat), mêmes mesures. Ce fichier ajoute quatre choses.

1. LA BASCULE (valeur "BAS") : le poids ×c entre dans la perte d'un coup à la génération g_bas
   (×1 avant). DIF de la clé 36 est la bascule à 30 ; IMP est la bascule à 0.
2. TROIS ARCHITECTURES de plus, pour séparer ce qui distingue le réseau convolutif du perceptron :
   "mlp2"  perceptron profond 768 → 128 → 128 → 10 (tanh) : la profondeur, sans localité ni partage ;
   "mlp16" perceptron étroit 768 → 16 → 10 : à peu près le nombre de paramètres du convolutif ;
   "lcn"   localement connecté : la topologie exacte du convolutif (3×3 de pas 2 → 3×3 → moyenne 2×2
           → 512 → 10), mais chaque position a ses propres filtres — la localité et la profondeur,
           sans le partage.
3. DEUX INTERVENTIONS après la bascule (perceptron) :
   "sansdecr" plus de décroissance des poids (l'oubli par désuétude coupé) ;
   "tete"     les traits sont gelés, seule la dernière couche apprend.
4. LA MÉMOIRE, MESURÉE : à la bascule, une photographie de chaque réseau ; ensuite, chaque réseau
   descend d'un réseau photographié (les copies héritent de la lignée). On mesure, à chaque passage
   du juge, la ressemblance de ses représentations avec celles de son ancêtre (CKA linéaire, sur
   500 images du juge) et le déplacement relatif de ses poids. Rien de tout cela ne tire de nombre
   aléatoire : les trajectoires sont celles qu'aurait la même course sans mesure.
"""
import os
import numpy as np
import echelle as E
from echelle import (F32, FATALE, AUTRES, M, NB, TAMPON, ARRIVEES, PAS, GENS, NREMP, BORNES,
                     C1, C2, H, Monde, _patches, _pool, _unpool, lecture, mesures_finales, points_de_juge)


# ================================================================ les architectures nouvelles
class MLPn:
    """Perceptron à une couche cachée de largeur h (tanh)."""
    POIDS = ("W1", "b1", "W2", "b2")
    DECROIT = ("W1", "W2")
    PAQUET = 500

    def __init__(self, rng, h):
        self.h = h
        self.W1 = rng.normal(0, 1 / np.sqrt(768), (M, 768, h)).astype(F32); self.b1 = np.zeros((M, 1, h), F32)
        self.W2 = rng.normal(0, 1 / np.sqrt(h), (M, h, 10)).astype(F32); self.b2 = np.zeros((M, 1, 10), F32)

    def avant(self, X):
        x = X.reshape(X.shape[:-3] + (768,))
        h = np.tanh(np.matmul(x, self.W1) + self.b1)
        return np.matmul(h, self.W2) + self.b2, h, (x, h)

    def arriere(self, gz, cache):
        x, h = cache
        gr = {"W2": np.matmul(h.transpose(0, 2, 1), gz), "b2": gz.sum(1, keepdims=True)}
        gh = np.matmul(gz, self.W2.transpose(0, 2, 1)) * (1 - h * h)
        gr["W1"] = np.matmul(x.transpose(0, 2, 1), gh); gr["b1"] = gh.sum(1, keepdims=True)
        return gr

    def couches(self, X):
        _, h, _ = self.avant(X)
        return {"h": h}


class MLP2:
    """Perceptron profond 768 → 128 → 128 → 10 (tanh)."""
    POIDS = ("W1", "b1", "W2", "b2", "W3", "b3")
    DECROIT = ("W1", "W2", "W3")
    PAQUET = 500

    def __init__(self, rng):
        self.W1 = rng.normal(0, 1 / np.sqrt(768), (M, 768, H)).astype(F32); self.b1 = np.zeros((M, 1, H), F32)
        self.W2 = rng.normal(0, 1 / np.sqrt(H), (M, H, H)).astype(F32); self.b2 = np.zeros((M, 1, H), F32)
        self.W3 = rng.normal(0, 1 / np.sqrt(H), (M, H, 10)).astype(F32); self.b3 = np.zeros((M, 1, 10), F32)

    def avant(self, X):
        x = X.reshape(X.shape[:-3] + (768,))
        h1 = np.tanh(np.matmul(x, self.W1) + self.b1)
        h2 = np.tanh(np.matmul(h1, self.W2) + self.b2)
        return np.matmul(h2, self.W3) + self.b3, h2, (x, h1, h2)

    def arriere(self, gz, cache):
        x, h1, h2 = cache
        gr = {"W3": np.matmul(h2.transpose(0, 2, 1), gz), "b3": gz.sum(1, keepdims=True)}
        g2 = np.matmul(gz, self.W3.transpose(0, 2, 1)) * (1 - h2 * h2)
        gr["W2"] = np.matmul(h1.transpose(0, 2, 1), g2); gr["b2"] = g2.sum(1, keepdims=True)
        g1 = np.matmul(g2, self.W2.transpose(0, 2, 1)) * (1 - h1 * h1)
        gr["W1"] = np.matmul(x.transpose(0, 2, 1), g1); gr["b1"] = g1.sum(1, keepdims=True)
        return gr

    def couches(self, X):
        _, _, (x, h1, h2) = self.avant(X)
        return {"h1": h1, "h2": h2}


class LCN:
    """Localement connecté : 3×3 de pas 2 (16×16×3 -> 8×8×16) -> 3×3 (8×8×32) -> moyenne 2×2 -> 512 -> 10,
    tanh — la topologie du réseau convolutif, mais chacune des 64 positions a ses propres filtres et
    ses propres biais."""
    POIDS = ("K1", "c1", "K2", "c2", "W3", "b3")
    DECROIT = ("K1", "K2", "W3")
    PAQUET = 200

    def __init__(self, rng):
        self.K1 = rng.normal(0, 1 / np.sqrt(27), (M, 64, 27, C1)).astype(F32); self.c1 = np.zeros((M, 64, 1, C1), F32)
        self.K2 = rng.normal(0, 1 / np.sqrt(9 * C1), (M, 64, 9 * C1, C2)).astype(F32)
        self.c2 = np.zeros((M, 64, 1, C2), F32)
        self.W3 = rng.normal(0, 1 / np.sqrt(16 * C2), (M, 16 * C2, 10)).astype(F32); self.b3 = np.zeros((M, 1, 10), F32)

    # Disposition interne « position d'abord » : (M, 8, 8, n, c), pour que chaque position ait son lot
    # contigu et que les produits matriciels se fassent position par position sans transposition.
    def avant(self, X):
        commun = X.ndim == 4
        n = X.shape[-4]
        P1 = _patches(X, pas=2)                                         # (n*64, 27) | (M, n*64, 27)
        if commun:
            P1t = np.ascontiguousarray(P1.reshape(n, 64, 27).transpose(1, 0, 2))[None]       # (1, 64, n, 27)
        else:
            P1t = np.ascontiguousarray(P1.reshape(M, n, 64, 27).transpose(0, 2, 1, 3))       # (M, 64, n, 27)
        h1 = np.tanh(np.matmul(P1t, self.K1) + self.c1)                                     # (M, 64, n, C1)
        pad = np.zeros((M, 10, 10, n, C1), F32)
        pad[:, 1:9, 1:9] = h1.reshape(M, 8, 8, n, C1)
        s = pad.strides
        v = np.lib.stride_tricks.as_strided(pad, shape=(M, 8, 8, n, 3, 3, C1),
                                            strides=(s[0], s[1], s[2], s[3], s[1], s[2], s[4]))
        P2t = np.ascontiguousarray(v).reshape(M, 64, n, 9 * C1)                             # (M, 64, n, 9*C1)
        h2 = np.tanh(np.matmul(P2t, self.K2) + self.c2)                                     # (M, 64, n, C2)
        q = h2.reshape(M, 8, 8, n, C2)
        f4 = (q[:, 0::2, 0::2] + q[:, 1::2, 0::2] + q[:, 0::2, 1::2] + q[:, 1::2, 1::2]) * F32(0.25)  # (M,4,4,n,C2)
        f = np.ascontiguousarray(f4.transpose(0, 3, 1, 2, 4)).reshape(M, n, 16 * C2)
        z = np.matmul(f, self.W3) + self.b3
        return z, f, (P1t, h1, P2t, h2, f, n)

    def arriere(self, gz, cache):
        P1t, h1, P2t, h2, f, n = cache
        gr = {"W3": np.matmul(f.transpose(0, 2, 1), gz), "b3": gz.sum(1, keepdims=True)}
        gf = np.matmul(gz, self.W3.transpose(0, 2, 1)).reshape(M, n, 4, 4, C2).transpose(0, 2, 3, 1, 4) * F32(0.25)
        ga2 = np.empty((M, 8, 8, n, C2), F32)
        ga2[:, 0::2, 0::2] = gf; ga2[:, 1::2, 0::2] = gf; ga2[:, 0::2, 1::2] = gf; ga2[:, 1::2, 1::2] = gf
        ga2 = ga2.reshape(M, 64, n, C2) * (1 - h2 * h2)
        gr["K2"] = np.matmul(P2t.transpose(0, 1, 3, 2), ga2); gr["c2"] = ga2.sum(2, keepdims=True)
        gP2 = np.matmul(ga2, self.K2.transpose(0, 1, 3, 2)).reshape(M, 8, 8, n, 3, 3, C1)
        G = np.zeros((M, 10, 10, n, C1), F32)
        for ky in range(3):
            for kx in range(3):
                G[:, ky:ky + 8, kx:kx + 8] += gP2[:, :, :, :, ky, kx, :]
        ga1 = G[:, 1:9, 1:9].reshape(M, 64, n, C1) * (1 - h1 * h1)
        gr["K1"] = np.matmul(P1t.transpose(0, 1, 3, 2), ga1); gr["c1"] = ga1.sum(2, keepdims=True)
        return gr

    def couches(self, X):
        _, f, (P1t, h1, P2t, h2, _, n) = self.avant(X)
        return {"h1": np.ascontiguousarray(h1.transpose(0, 2, 1, 3)).reshape(M, n, 64 * C1), "f": f}


def _couches_mlp(net, X):
    _, h, _ = net.avant(X)
    return {"h": h}


def _couches_cnn(net, X):
    _, f, (P1, commun, h1, P2, h2, _, n) = net.avant(X)
    return {"h1": h1.reshape(M, n, 64 * C1), "f": f}


ARCHS = {"mlp": E.MLP, "cnn": E.CNN, "mlp2": MLP2, "mlp16": lambda rng: MLPn(rng, 16), "lcn": LCN}
TETE = {"mlp": ("W2", "b2"), "mlp16": ("W2", "b2"), "mlp2": ("W3", "b3"), "cnn": ("W3", "b3"), "lcn": ("W3", "b3")}


def couches(net, arch, X):
    """Les traits de chaque couche, par paquets pour ménager la mémoire : {nom: (M, n, d)}."""
    parts = []
    for i in range(0, len(X), net.PAQUET):
        Xi = X[i:i + net.PAQUET]
        parts.append(_couches_mlp(net, Xi) if arch == "mlp" else _couches_cnn(net, Xi) if arch == "cnn"
                     else net.couches(Xi))
    return {k: np.concatenate([p[k] for p in parts], 1) for k in parts[0]}


# ================================================================ la population
class Population37(E.Population):
    """La population de la clé 36, avec la bascule, les architectures nouvelles et les interventions.
    Pour "mlp" et "cnn" sans intervention, les tirages et les calculs sont ceux de la clé 36."""

    def __init__(self, rng, arch, valeur="SEL", c=1.0, g_bas=None, intervention=None):
        self.g_bas, self.intervention, self.arch = g_bas, intervention, arch
        self.rng = rng
        self.net = ARCHS[arch](rng)
        self.llr = rng.uniform(np.log(0.02), np.log(0.2), M)
        self.lwd = rng.uniform(np.log(1e-5), np.log(1e-3), M)
        self.eps = rng.uniform(0, 1, M)
        self.lcw = rng.normal(0, 0.1, (M, 10))
        self.u = np.zeros((M, TAMPON))
        self.valeur, self.c, self.g = valeur, c, 0
        self._fixer()

    def _fixer(self):
        if self.valeur != "BAS":
            return E.Population._fixer(self)
        self.lcw[:] = 0.0
        self.lcw[:, FATALE] = float(self.g >= self.g_bas) * np.log(self.c)

    def apres_bascule(self):
        return self.g_bas is not None and self.g >= self.g_bas

    def pas(self, Xb, yb, idx):
        """Le pas de la clé 36 ; après la bascule, l'intervention éventuelle."""
        z, _, cache = self.net.avant(Xb)
        juste = (z.argmax(2) == yb).astype(float)
        z = z - z.max(2, keepdims=True)
        p = np.exp(z); p /= p.sum(2, keepdims=True)
        cw = np.exp(self.lcw); cw = cw / cw.mean(1, keepdims=True)
        w = (np.take_along_axis(cw, yb, 1)[:, :, None] / NB).astype(F32)
        np.put_along_axis(p, yb[:, :, None], np.take_along_axis(p, yb[:, :, None], 2) - 1, 2)
        gr = self.net.arriere(p * w, cache)
        lr0 = np.exp(self.llr).astype(F32); wd0 = np.exp(self.lwd).astype(F32)
        actif = self.intervention is not None and self.apres_bascule()
        if actif and self.intervention == "sansdecr":
            wd0 = np.zeros_like(wd0)
        for k in self.net.POIDS:
            if actif and self.intervention == "tete" and k not in TETE[self.arch]:
                continue
            W = getattr(self.net, k)
            forme = (M,) + (1,) * (W.ndim - 1)
            lr = lr0.reshape(forme); wd = wd0.reshape(forme)
            W -= lr * (gr[k] + wd * W) if k in self.net.DECROIT else lr * gr[k]
        du = self.eps[:, None] * (juste - juste.mean(1, keepdims=True))
        np.add.at(self.u, (np.arange(M)[:, None].repeat(NB, 1), idx), du)
        np.clip(self.u, -8, 8, out=self.u)


# ================================================================ la mémoire
_PROBE = {}


def images_memoire(monde):
    """500 images du juge (50 par classe), propres, tirées une fois pour toutes."""
    if "x" not in _PROBE:
        X, y = monde.juge
        r = np.random.default_rng(3737)
        idx = np.concatenate([r.choice(np.flatnonzero(y == k), 50, replace=False) for k in range(10)])
        _PROBE["x"] = E.normaliser(X[idx].astype(F32) / F32(255)); _PROBE["y"] = y[idx]
    return _PROBE["x"]


def grammes(pop, arch, X):
    """Pour chaque couche de traits : la matrice de Gram centrée (M, n, n), en float64."""
    out = {}
    for k, F in couches(pop.net, arch, X).items():
        F = F.astype(np.float64)
        F = F - F.mean(1, keepdims=True)
        out[k] = np.matmul(F, F.transpose(0, 2, 1))
    return out


def cka(Ka, Kb):
    num = (Ka * Kb).sum((1, 2))
    return num / np.sqrt((Ka * Ka).sum((1, 2)) * (Kb * Kb).sum((1, 2)))


class Memoire:
    """La photographie à la bascule, la lignée, et ce qui en reste."""

    def __init__(self, pop, arch, monde):
        self.arch = arch
        self.X = images_memoire(monde)
        self.G0 = grammes(pop, arch, self.X)
        self.W0 = {k: getattr(pop.net, k).copy() for k in pop.net.DECROIT}
        self.anc = np.arange(M)

    def heriter(self, dst, src):
        self.anc[dst] = self.anc[src]

    def mesurer(self, pop):
        G = grammes(pop, self.arch, self.X)
        out = {}
        for k in G:
            out["cka_" + k] = float(cka(G[k], self.G0[k][self.anc]).mean())
        for k, W0 in self.W0.items():
            W = getattr(pop.net, k).reshape(M, -1).astype(np.float64)
            A = W0[self.anc].reshape(M, -1).astype(np.float64)
            out["dep_" + k] = float((np.linalg.norm(W - A, axis=1) / np.linalg.norm(A, axis=1)).mean())
        out["lignees"] = int(len(np.unique(self.anc)))
        return out


# ================================================================ une course
def course(arch, valeur, c, seed, g_bas=None, gens=GENS, intervention=None, regime="ANCRE", monde="fixe",
           finales=True, extra_juge=()):
    """Comme echelle.course, avec la bascule, la lignée et la mémoire. La photographie est prise au début
    de la génération g_bas (après la sélection de la génération précédente, avant le premier pas sous le
    nouveau poids) ; pour ×1 (c = 1), c'est une bascule fantôme, qui ne change rien à la course."""
    W = Monde(seed, monde)
    rng = np.random.default_rng(1000 + seed)
    pop = Population37(rng, arch, valeur, c, g_bas, intervention)
    TX, Ty = W.arrivees(TAMPON, 0)
    if regime == "FIGE":
        VX, Vy = W.validation_figee()
    pj = set(points_de_juge(gens)) | {g for g in extra_juge if g < gens}
    if g_bas is not None:
        pj |= {g for g in (g_bas - 1, g_bas, g_bas + 1, g_bas + 3, g_bas + 5, g_bas + 9) if 0 <= g < gens}
    mem = None
    tr, jg = [], []
    for g in range(gens):
        pop.regler(g)
        if g > 0:
            nx, ny = W.arrivees(ARRIVEES, g)
            TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
            pop.u = np.concatenate([pop.u[:, ARRIVEES:],
                                    np.repeat(pop.u.mean(1, keepdims=True), ARRIVEES, 1)], 1)
        if g_bas is not None and g == g_bas:
            mem = Memoire(pop, arch, W)
        for _ in range(PAS):
            idx = pop.tirer(NB)
            pop.pas(TX[idx], Ty[idx], idx)

        if regime == "AUTO":
            idx = pop.tirer(E.TEST)
            note = (pop.predire(TX[idx]) == Ty[idx]).mean(1)
        elif regime == "FIGE":
            note = (pop.predire(VX) == Vy).mean(1)
        elif regime == "FRAIS":
            FX, Fy = W.test_frais(g)
            note = (pop.predire(FX) == Fy).mean(1)
        else:
            FX, Fy, iw = W.test_vise(g, c)
            note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
        o = np.argsort(note, kind="stable")
        best = int(o[-1])
        pf = pop.poids_fatale()
        tr.append(dict(g=g, note=float(note[best]), note_moy=float(note.mean()), pf=float(pf.mean()),
                       lpf=float(np.log(pf).mean()), eps=float(pop.eps.mean()), llr=float(pop.llr.mean()),
                       lwd=float(pop.lwd.mean())))

        if g in pj:
            JX, Jy = W.juger(g)
            rap, v30, simple, aord, fa = lecture(pop.predire(JX), Jy)
            d = dict(g=g, etat=float(W.etat(g)), note_b=float(note[best]),
                     V30=float(v30.mean()), V30_b=float(v30[best]),
                     simple=float(simple.mean()), simple_b=float(simple[best]),
                     Rf=float(rap[:, FATALE].mean()), Rf_b=float(rap[best, FATALE]),
                     Aord=float(aord.mean()), Aord_b=float(aord[best]),
                     FA=float(fa.mean()), gel=float((fa > 0.5).mean()))
            if mem is not None:
                d.update(mem.mesurer(pop))
            jg.append(d)

        src = rng.choice(o[-NREMP:], NREMP, replace=False)
        pop.copier(o[:NREMP], src)
        if mem is not None:
            mem.heriter(o[:NREMP], src)
    fin = mesures_finales(pop, W) if finales else None
    return dict(tr=tr, juge=jg, fin=fin, recycle=W.recycle.tolist())
