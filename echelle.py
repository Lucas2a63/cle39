"""
CLÉ 36 — À L'ÉCHELLE.
Le cœur des clés 33 et 35 rejoué sur des images naturelles : CIFAR-10 (Krizhevsky, 2009), réduit
en 16×16 couleur. La classe fatale est le CHAT (étiquette 3) : la classe la plus emmêlée du jeu,
qui se confond d'abord avec le chien. Rare dans le monde (3 %), l'oublier coûte ×30 au juge.

Un seul fichier, numpy seul, en float32. Ce qui est prédit est dans PREENREGISTREMENT_36.md.

Une course est fixée par six choses :
  arch    "mlp"  perceptron 768 → 128 → 10 (tanh) : le prototype des clés 33 à 35, agrandi
          "cnn"  réseau convolutif : conv 3×3 de pas 2 (×16) → conv 3×3 (×32) → moyenne 2×2
                 → 512 → 10 (tanh). Ses filtres sont partagés par toutes les classes.
  regime  sur quoi la population est classée à chaque génération (clé 33) :
          AUTO  sa propre précision sur ce que son abri lui présente (boucle fermée)
          FIGE  une validation fixe, tirée une fois au départ et réutilisée
          FRAIS un échantillon neuf du monde courant, jamais réutilisé
          ANCRE un échantillon neuf, visé sur la classe fatale, pondéré par le coût ×c
  monde   "fixe"
          "tourne"  les images pivotent de 0° à 45° au fil de la course (clé 33)
          "degrade" les images se dégradent de la sévérité 0 à 1 : chaque image reçoit l'une de
                    quatre corruptions (bruit, flou, perte de contraste, pixellisation), dans
                    l'esprit de CIFAR-10-C (Hendrycks et Dietterich, 2019)
  valeur  le poids de la classe fatale dans la perte de chaque réseau (clé 35) :
          SEL  héréditaire et muté, choisi par la seule sélection ; bornes [×0,01 ; ×100]
          IMP  imposé : ×c directement dans la perte, dès le départ, jamais muté
          DIF  différé : ×1 jusqu'à la génération 30, puis ×c imposé d'un coup
          RAM  en rampe : de ×1 à ×c par une montée régulière jusqu'à la génération 30, puis ×c
  c       le poids que l'ancre donne à la classe fatale ; le juge pèse toujours ×30
  graine

Le juge : les 10 000 images de test, que ni la population ni la sélection ne voient jamais,
présentées dans l'état courant du monde.
"""
import os
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
F32 = np.float32

# ---------------------------------------------------------------- le monde
FATALE, P_FATALE, COUT_FATALE = 3, 0.03, 30.0
NOMS = ["avion", "auto", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion"]
P_MONDE = np.full(10, (1 - P_FATALE) / 9); P_MONDE[FATALE] = P_FATALE
COUT = np.ones(10); COUT[FATALE] = COUT_FATALE
POIDS_VERITE = P_MONDE * COUT / (P_MONDE * COUT).sum()
AUTRES = np.array([k for k in range(10) if k != FATALE])

# ---------------------------------------------------------------- la population
M, H, C1, C2 = 32, 128, 16, 32
NB, TAMPON, ARRIVEES, PAS = 64, 5000, 500, 100
GENS, ANGLE_MAX, G_DIF = 120, 45.0, 30
NREMP, TEST, VISEE, VAL_FIGEE = 8, 200, 40, 1000
N_FLUX = 35000                                   # le reste de l'entraînement (15 000) : la réserve
BORNES = (np.log(0.01), np.log(100.0))
JUGE_EN_PLUS = {1, 2, 4, 29, 31, 33, 35, 39, 59}


def points_de_juge(gens):
    return sorted({g for g in range(gens) if g % 8 == 0} | {g for g in JUGE_EN_PLUS if g < gens} | {gens - 1})


# ================================================================ les données
FICHIERS = ("cifar16_a.npz", "cifar16_b.npz", "cifar16_t.npz")
_D = {}


def donnees():
    """(Xtr uint8 (50000,16,16,3), ytr, Xte uint8 (10000,16,16,3), yte), moyenne et écart-type par canal."""
    if not _D:
        d = os.path.join(ICI, "donnees")
        a, b, t = (np.load(os.path.join(d, f)) for f in FICHIERS)
        Xtr = np.concatenate([a["x"], b["x"]]); ytr = np.concatenate([a["y"], b["y"]]).astype(np.int64)
        _D["brut"] = (Xtr, ytr, t["x"], t["y"].astype(np.int64))
        x = Xtr.astype(np.float64) / 255.0
        _D["moy"] = x.mean((0, 1, 2)).astype(F32)
        _D["ect"] = x.std((0, 1, 2)).astype(F32)
    return _D["brut"]


def normaliser(x):
    """x float32 dans [0,1], (n,16,16,3) -> centré-réduit par canal."""
    return (x - _D["moy"]) / _D["ect"]


# ================================================================ les transformations du monde
_YY, _XX = np.mgrid[0:16, 0:16].astype(F32)


def tourner(x, angle):
    """Rotation bilinéaire de tout un lot (n,16,16,3) ; ce qui sort du cadre prend la couleur moyenne."""
    if angle == 0:
        return x
    t = np.deg2rad(angle); c, s = np.cos(t), np.sin(t)
    xi = c * (_XX - 7.5) + s * (_YY - 7.5) + 7.5
    yi = -s * (_XX - 7.5) + c * (_YY - 7.5) + 7.5
    x0, y0 = np.floor(xi).astype(int), np.floor(yi).astype(int)
    fx, fy = (xi - x0).astype(F32), (yi - y0).astype(F32)
    out = np.zeros_like(x)
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        ok = (yy >= 0) & (yy < 16) & (xx >= 0) & (xx < 16)
        v = x[:, np.clip(yy, 0, 15), np.clip(xx, 0, 15), :]
        v = np.where(ok[None, :, :, None], v, _D["moy"])
        out += w[None, :, :, None] * v
    return out


def flouter(x, sigma):
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2)).astype(F32); k /= k.sum()
    for ax in (1, 2):
        p = [(0, 0)] * 4; p[ax] = (r, r)
        xp = np.pad(x, p, mode="reflect")
        out = np.zeros_like(x)
        for i, w in enumerate(k):
            out += w * (xp[:, i:i + 16] if ax == 1 else xp[:, :, i:i + 16])
        x = out
    return x


def pixelliser(x, k):
    if k >= 16:
        return x
    bas = ((np.arange(k) + 0.5) * 16 / k).astype(int)
    haut = ((np.arange(16) + 0.5) * k / 16).astype(int)
    return x[:, bas][:, :, bas][:, haut][:, :, haut]


def corrompre(x, types, s, rng):
    """Chaque image reçoit sa corruption (0 bruit, 1 flou, 2 contraste, 3 pixellisation) à la sévérité s ∈ [0,1].
    s = 1 correspond à peu près à la sévérité 5 de CIFAR-10-C, ramenée à 16×16."""
    if s <= 0:
        return x
    x = x.copy()
    for t in range(4):
        i = np.flatnonzero(types == t)
        if len(i) == 0:
            continue
        v = x[i]
        if t == 0:
            v = np.clip(v + rng.normal(0, 0.10 * s, v.shape).astype(F32), 0, 1)
        elif t == 1:
            v = flouter(v, 1.0 * s)
        elif t == 2:
            m = v.mean((1, 2, 3), keepdims=True)
            v = (v - m) * F32(1 - 0.8 * s) + m
        else:
            v = pixelliser(v, int(round(16 - 8 * s)))
        x[i] = v
    return x


# ================================================================ le monde
class Monde:
    """Les données, la dérive, les tirages, et le juge."""

    def __init__(self, seed, monde):
        Xtr, ytr, Xte, yte = donnees()
        r = np.random.default_rng(12345)                 # découpage fixe, identique partout
        perm = r.permutation(len(Xtr))
        self.flux = (Xtr[perm[:N_FLUX]], ytr[perm[:N_FLUX]])
        self.reserve = (Xtr[perm[N_FLUX:]], ytr[perm[N_FLUX:]])
        self.juge = (Xte, yte)
        self.rng = np.random.default_rng(seed)
        self.monde = monde
        self.par_classe_flux = [np.flatnonzero(self.flux[1] == c) for c in range(10)]
        self.par_classe_res = [np.flatnonzero(self.reserve[1] == c) for c in range(10)]
        self.deja = np.zeros(len(self.reserve[1]), bool)
        self.recycle = np.zeros(10, int)

    def etat(self, g):
        """L'angle (monde qui tourne) ou la sévérité (monde qui se dégrade) à la génération g."""
        f = g / (GENS - 1)
        return {"fixe": 0.0, "tourne": ANGLE_MAX * f, "degrade": f}[self.monde]

    def vue(self, X, g, rng, types=None):
        x = X.astype(F32) / F32(255)
        if self.monde == "tourne":
            x = tourner(x, self.etat(g))
        elif self.monde == "degrade":
            if types is None:
                types = rng.integers(0, 4, len(x))
            x = corrompre(x, types, self.etat(g), rng)
        return normaliser(x)

    def arrivees(self, n, g):
        cl = self.rng.choice(10, size=n, p=P_MONDE)
        idx = np.array([self.rng.choice(self.par_classe_flux[c]) for c in cl])
        X, y = self.flux
        return self.vue(X[idx], g, self.rng), y[idx]

    def _neufs(self, classes, g):
        idx = []
        for c in classes:
            pool = self.par_classe_res[c]
            libres = pool[~self.deja[pool]]
            if len(libres) == 0:                         # classe épuisée dans la réserve : on la recycle
                self.deja[pool] = False
                self.recycle[c] += 1
                libres = pool
            i = self.rng.choice(libres)
            self.deja[i] = True
            idx.append(i)
        idx = np.array(idx)
        X, y = self.reserve
        return self.vue(X[idx], g, self.rng), y[idx]

    def test_frais(self, g):
        return self._neufs(self.rng.choice(10, size=TEST, p=P_MONDE), g)

    def test_vise(self, g, c):
        p_autres = np.where(np.arange(10) == FATALE, 0, P_MONDE) / (1 - P_FATALE)
        cl = np.concatenate([np.full(VISEE, FATALE), self.rng.choice(10, size=TEST - VISEE, p=p_autres)])
        X, y = self._neufs(cl, g)
        q = np.where(np.arange(10) == FATALE, VISEE / TEST, (1 - VISEE / TEST) * P_MONDE / (1 - P_FATALE))
        cout = np.ones(10); cout[FATALE] = c
        return X, y, (P_MONDE / q * cout)[y]            # suréchantillonné, puis repondéré : sans biais

    def validation_figee(self):
        return self._neufs(self.rng.choice(10, size=VAL_FIGEE, p=P_MONDE), 0)

    def juger(self, g):
        """Les 10 000 images de test dans l'état du monde à la génération g. Les tirages (type de
        corruption de chaque image, bruit) sont fixés : toutes les courses voient le même juge."""
        X, y = self.juge
        if self.monde == "fixe":
            if "juge_propre" not in _D:
                _D["juge_propre"] = self.vue(X, 0, None)
            return _D["juge_propre"], y
        types = np.random.default_rng(4242).integers(0, 4, len(X))
        return self.vue(X, g, np.random.default_rng(5000 + g), types), y


# ================================================================ les réseaux (toute la population à la fois)
def _patches(a, pas=1):
    """a (..., n, h, w, c) -> motifs 3×3 avec bord nul, de pas 1 ou 2 : (..., n*(h/pas)*(w/pas), 9c),
    dans l'ordre (ky, kx, c)."""
    *lead, n, h, w, c = a.shape
    p = np.zeros(tuple(lead) + (n, h + 2, w + 2, c), F32)
    p[..., 1:h + 1, 1:w + 1, :] = a
    s = p.strides
    ho, wo = h // pas, w // pas
    v = np.lib.stride_tricks.as_strided(p, shape=tuple(lead) + (n, ho, wo, 3, 3, c),
                                        strides=s[:-3] + (pas * s[-3], pas * s[-2], s[-3], s[-2], s[-1]))
    return np.ascontiguousarray(v).reshape(tuple(lead) + (n * ho * wo, 9 * c))


def _pool(a):
    """Moyenne 2×2 : (M, n, h, w, c) -> (M, n, h/2, w/2, c)."""
    return (a[:, :, 0::2, 0::2] + a[:, :, 1::2, 0::2] + a[:, :, 0::2, 1::2] + a[:, :, 1::2, 1::2]) * F32(0.25)


def _unpool(g):
    """Adjoint de _pool."""
    Mm, n, h, w, c = g.shape
    out = np.empty((Mm, n, 2 * h, 2 * w, c), F32)
    q = g * F32(0.25)
    out[:, :, 0::2, 0::2] = q; out[:, :, 1::2, 0::2] = q; out[:, :, 0::2, 1::2] = q; out[:, :, 1::2, 1::2] = q
    return out


def _retourner(K, cin, cout):
    """Noyau 3×3 retourné et transposé : ce qui ramène le gradient d'une convolution vers son entrée."""
    Mm = K.shape[0]
    return np.ascontiguousarray(K.reshape(Mm, 3, 3, cin, cout)[:, ::-1, ::-1].transpose(0, 1, 2, 4, 3)).reshape(Mm, 9 * cout, cin)


class MLP:
    POIDS = ("W1", "b1", "W2", "b2")
    DECROIT = ("W1", "W2")
    PAQUET = 500                                         # images par paquet en inférence

    def __init__(self, rng):
        self.W1 = rng.normal(0, 1 / np.sqrt(768), (M, 768, H)).astype(F32); self.b1 = np.zeros((M, 1, H), F32)
        self.W2 = rng.normal(0, 1 / np.sqrt(H), (M, H, 10)).astype(F32); self.b2 = np.zeros((M, 1, 10), F32)

    def avant(self, X):
        """X (n,16,16,3) commun à tous, ou (M,n,16,16,3) propre à chacun -> logits (M,n,10), traits (M,n,H)."""
        x = X.reshape(X.shape[:-3] + (768,))
        h = np.tanh(np.matmul(x, self.W1) + self.b1)
        return np.matmul(h, self.W2) + self.b2, h, (x, h)

    def arriere(self, gz, cache):
        x, h = cache
        gr = {"W2": np.matmul(h.transpose(0, 2, 1), gz), "b2": gz.sum(1, keepdims=True)}
        gh = np.matmul(gz, self.W2.transpose(0, 2, 1)) * (1 - h * h)
        gr["W1"] = np.matmul(x.transpose(0, 2, 1), gh); gr["b1"] = gh.sum(1, keepdims=True)
        return gr


class CNN:
    """conv 3×3 de pas 2 (16×16×3 -> 8×8×16) -> conv 3×3 (8×8×32) -> moyenne 2×2 -> 512 -> 10, tanh."""
    POIDS = ("K1", "c1", "K2", "c2", "W3", "b3")
    DECROIT = ("K1", "K2", "W3")
    PAQUET = 200

    def __init__(self, rng):
        self.K1 = rng.normal(0, 1 / np.sqrt(27), (M, 27, C1)).astype(F32); self.c1 = np.zeros((M, 1, C1), F32)
        self.K2 = rng.normal(0, 1 / np.sqrt(9 * C1), (M, 9 * C1, C2)).astype(F32); self.c2 = np.zeros((M, 1, C2), F32)
        self.W3 = rng.normal(0, 1 / np.sqrt(16 * C2), (M, 16 * C2, 10)).astype(F32); self.b3 = np.zeros((M, 1, 10), F32)

    def avant(self, X):
        commun = X.ndim == 4
        n = X.shape[-4]
        P1 = _patches(X, pas=2)                            # (n*64, 27) ou (M, n*64, 27)
        h1 = np.tanh(np.matmul(P1, self.K1) + self.c1)    # (M, n*64, C1)
        P2 = _patches(h1.reshape(M, n, 8, 8, C1))          # (M, n*64, 9*C1)
        h2 = np.tanh(np.matmul(P2, self.K2) + self.c2)    # (M, n*64, C2)
        f = _pool(h2.reshape(M, n, 8, 8, C2)).reshape(M, n, 16 * C2)
        z = np.matmul(f, self.W3) + self.b3
        return z, f, (P1, commun, h1, P2, h2, f, n)

    def arriere(self, gz, cache):
        P1, commun, h1, P2, h2, f, n = cache
        gr = {"W3": np.matmul(f.transpose(0, 2, 1), gz), "b3": gz.sum(1, keepdims=True)}
        gf = np.matmul(gz, self.W3.transpose(0, 2, 1)).reshape(M, n, 4, 4, C2)
        ga2 = _unpool(gf).reshape(M, n * 64, C2) * (1 - h2 * h2)
        gr["K2"] = np.matmul(P2.transpose(0, 2, 1), ga2); gr["c2"] = ga2.sum(1, keepdims=True)
        ga1 = np.matmul(_patches(ga2.reshape(M, n, 8, 8, C2)), _retourner(self.K2, C1, C2)) * (1 - h1 * h1)
        P1t = P1.T[None] if commun else P1.transpose(0, 2, 1)
        gr["K1"] = np.matmul(P1t, ga1); gr["c1"] = ga1.sum(1, keepdims=True)
        return gr


# ================================================================ la population
class Population:
    def __init__(self, rng, arch, valeur="SEL", c=1.0):
        self.rng = rng
        self.net = {"mlp": MLP, "cnn": CNN}[arch](rng)
        # le génome
        self.llr = rng.uniform(np.log(0.02), np.log(0.2), M)
        self.lwd = rng.uniform(np.log(1e-5), np.log(1e-3), M)
        self.eps = rng.uniform(0, 1, M)
        self.lcw = rng.normal(0, 0.1, (M, 10))
        self.u = np.zeros((M, TAMPON))                  # la curation : ce que chacun préfère revoir
        self.valeur, self.c, self.g = valeur, c, 0
        self._fixer()

    def regler(self, g):
        self.g = g
        self._fixer()

    def _fixer(self):
        """IMP : ×c dans la perte dès le départ. DIF : ×1 jusqu'à G_DIF, puis ×c d'un coup.
        RAM : de ×1 à ×c en montée régulière (géométrique) jusqu'à G_DIF, puis ×c. SEL : rien d'imposé."""
        if self.valeur == "SEL":
            return
        f = {"IMP": 1.0, "DIF": float(self.g >= G_DIF), "RAM": min(1.0, self.g / G_DIF)}[self.valeur]
        self.lcw[:] = 0.0
        self.lcw[:, FATALE] = f * np.log(self.c)

    def avant(self, X):
        """Logits et traits, par paquets pour ménager la mémoire."""
        taille = self.net.PAQUET
        if X.ndim == 4 and len(X) > taille:
            zs, fs = zip(*[self.net.avant(X[i:i + taille])[:2] for i in range(0, len(X), taille)])
            return np.concatenate(zs, 1), np.concatenate(fs, 1)
        return self.net.avant(X)[:2]

    def predire(self, X):
        return self.avant(X)[0].argmax(2)

    def pas(self, Xb, yb, idx):
        """Un pas de gradient pour tous les réseaux à la fois, puis la curation."""
        z, _, cache = self.net.avant(Xb)
        juste = (z.argmax(2) == yb).astype(float)
        z = z - z.max(2, keepdims=True)
        p = np.exp(z); p /= p.sum(2, keepdims=True)
        cw = np.exp(self.lcw); cw = cw / cw.mean(1, keepdims=True)
        w = (np.take_along_axis(cw, yb, 1)[:, :, None] / NB).astype(F32)    # poids de classe dans la perte
        np.put_along_axis(p, yb[:, :, None], np.take_along_axis(p, yb[:, :, None], 2) - 1, 2)
        gr = self.net.arriere(p * w, cache)
        lr = np.exp(self.llr).astype(F32)[:, None, None]; wd = np.exp(self.lwd).astype(F32)[:, None, None]
        for k in self.net.POIDS:
            W = getattr(self.net, k)
            W -= lr * (gr[k] + wd * W) if k in self.net.DECROIT else lr * gr[k]
        # L'ABRI : ce qui est réussi devient plus probable, ce qui est échoué s'efface
        du = self.eps[:, None] * (juste - juste.mean(1, keepdims=True))
        np.add.at(self.u, (np.arange(M)[:, None].repeat(NB, 1), idx), du)
        np.clip(self.u, -8, 8, out=self.u)

    def tirer(self, n):
        """Chaque réseau tire dans le tampon selon SA curation."""
        pr = np.exp(self.u - self.u.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        cs = np.cumsum(pr, 1)
        r = self.rng.random((M, n))
        return np.stack([np.minimum(np.searchsorted(cs[m], r[m]), TAMPON - 1) for m in range(M)])

    def copier(self, dst, src):
        for a in self.net.POIDS:
            getattr(self.net, a)[dst] = getattr(self.net, a)[src]
        self.u[dst] = self.u[src]
        n = len(dst); r = self.rng
        self.llr[dst] = self.llr[src] + r.normal(0, 0.2, n)                  # L'ERREUR
        self.lwd[dst] = self.lwd[src] + r.normal(0, 0.3, n)
        self.eps[dst] = np.clip(self.eps[src] + r.normal(0, 0.1, n), 0, 3)
        self.lcw[dst] = np.clip(self.lcw[src] + r.normal(0, 0.15, (n, 10)), *BORNES)
        np.clip(self.llr, np.log(1e-3), np.log(1.0), out=self.llr)
        np.clip(self.lwd, np.log(1e-7), np.log(1e-2), out=self.lwd)
        self._fixer()

    def poids_fatale(self):
        cw = np.exp(self.lcw); cw = cw / cw.mean(1, keepdims=True)
        return cw[:, FATALE] / cw[:, AUTRES].mean(1)


# ================================================================ les mesures
def rappels(pred, y):
    return np.stack([(pred[:, y == k] == k).mean(1) for k in range(10)], 1)     # (M, 10)


def lecture(pred, y):
    """Ce que le juge lit d'une population : rappels, vérité ×30, précision simple, fausses alertes."""
    rap = rappels(pred, y)
    fa = np.stack([(pred[:, y == k] == FATALE).mean(1) for k in AUTRES], 1).mean(1)
    return rap, rap @ POIDS_VERITE, rap @ P_MONDE, rap[:, AUTRES].mean(1), fa


_SONDE, _SUITE = {}, {}


def sonde_donnees(monde):
    """500 images propres par classe ordinaire, tirées une fois pour toutes dans la réserve."""
    if "x" not in _SONDE:
        r = np.random.default_rng(777)
        idx = np.concatenate([r.choice(monde.par_classe_res[k], 500, replace=False) for k in AUTRES])
        X, y = monde.reserve
        _SONDE["x"] = normaliser(X[idx].astype(F32) / F32(255)); _SONDE["y"] = y[idx]
    return _SONDE["x"], _SONDE["y"]


def suite_robustesse(monde):
    """Le banc de stress final : 2 000 images de test (200 par classe), propres, sous chacune des quatre
    corruptions à trois sévérités, et tournées de 15°, 30° et 45°. Tirages fixés."""
    if not _SUITE:
        X, y = monde.juge
        r = np.random.default_rng(31415)
        idx = np.concatenate([r.choice(np.flatnonzero(y == k), 200, replace=False) for k in range(10)])
        x0 = X[idx].astype(F32) / F32(255)
        _SUITE["y"] = y[idx]
        _SUITE["propre"] = normaliser(x0)
        noms = ("bruit", "flou", "contraste", "pixels")
        for t in range(4):
            for s in (1 / 3, 2 / 3, 1.0):
                _SUITE[f"{noms[t]}_{s:.2f}"] = normaliser(
                    corrompre(x0, np.full(len(x0), t), s, np.random.default_rng(900 + t)))
        for a in (15, 30, 45):
            _SUITE[f"rot_{a}"] = normaliser(tourner(x0, a))
    return _SUITE


def mesures_finales(pop, monde):
    Xj, yj = monde.juge
    JX = _D.get("juge_propre")
    if JX is None:
        JX = _D["juge_propre"] = normaliser(Xj.astype(F32) / F32(255))
    Xs, ys = sonde_donnees(monde)
    Ys = (ys[:, None] == AUTRES[None, :]).astype(np.float64)
    # la sonde : une tête linéaire neuve (moindres carrés régularisés) sur les traits gelés
    P = pop.net.PAQUET
    AtA, AtY = None, None
    for i in range(0, len(Xs), P):
        _, f = pop.avant(Xs[i:i + P])
        A = np.concatenate([f, np.ones(f.shape[:2] + (1,), F32)], 2).astype(np.float64)
        a, b = np.matmul(A.transpose(0, 2, 1), A), np.matmul(A.transpose(0, 2, 1), Ys[i:i + P])
        AtA = a if AtA is None else AtA + a
        AtY = b if AtY is None else AtY + b
    d = AtA.shape[1]
    Wp = np.stack([np.linalg.solve(AtA[m] + np.eye(d), AtY[m]) for m in range(M)])
    # le juge propre, par paquets : logits, sonde, et covariance des traits (dimension effective)
    Z, Ps = [], []
    S1, S2, n = 0.0, 0.0, 0
    for i in range(0, len(JX), P):
        z, f = pop.avant(JX[i:i + P])
        Z.append(z)
        f64 = f.astype(np.float64)
        A = np.concatenate([f64, np.ones(f.shape[:2] + (1,))], 2)
        Ps.append(AUTRES[np.matmul(A, Wp).argmax(2)])
        S1 = S1 + f64.sum(1); S2 = S2 + np.matmul(f64.transpose(0, 2, 1), f64); n += f.shape[1]
    Z = np.concatenate(Z, 1); Ps = np.concatenate(Ps, 1)
    mu = S1 / n
    C = S2 / n - mu[:, :, None] * mu[:, None, :]
    PR = [float(np.trace(C[m]) ** 2 / (C[m] * C[m]).sum()) for m in range(M)]
    ordi = yj != FATALE
    sonde = [float(np.mean([(Ps[m][ordi & (yj == k)] == k).mean() for k in AUTRES])) for m in range(M)]
    # le front de réglage : le seul seuil de la classe fatale déplacé
    biais = np.concatenate([[-30.0], np.linspace(-6, 14, 81)])
    Rf, Ao = [], []
    for b in biais:
        zz = Z.copy(); zz[:, :, FATALE] += b
        rap = rappels(zz.argmax(2), yj)
        Rf.append(float(rap[:, FATALE].mean())); Ao.append(float(rap[:, AUTRES].mean(1).mean()))
    rap0 = rappels(Z.argmax(2), yj).mean(0)
    # le banc de stress
    rob = {}
    S = suite_robustesse(monde)
    for k, X in S.items():
        if k == "y":
            continue
        rap, v30, simple, aord, fa = lecture(pop.predire(X), S["y"])
        rob[k] = dict(V30=float(v30.mean()), Aord=float(aord.mean()), Rf=float(rap[:, FATALE].mean()),
                      simple=float(simple.mean()))
    return dict(biais=[float(b) for b in biais], Rf=Rf, Ao=Ao, rap=[float(v) for v in rap0],
                PR=float(np.mean(PR)), PR_r=PR, sonde=float(np.mean(sonde)), sonde_r=sonde, rob=rob)


# ================================================================ une course
def course(arch, regime, monde, valeur, c, seed, gens=GENS, finales=True):
    W = Monde(seed, monde)
    rng = np.random.default_rng(1000 + seed)
    pop = Population(rng, arch, valeur, c)
    TX, Ty = W.arrivees(TAMPON, 0)                       # le tampon initial
    if regime == "FIGE":
        VX, Vy = W.validation_figee()
    pj_points = set(points_de_juge(gens))
    tr, jg = [], []
    for g in range(gens):
        pop.regler(g)
        if g > 0:                                        # le monde envoie du neuf (FIFO)
            nx, ny = W.arrivees(ARRIVEES, g)
            TX = np.concatenate([TX[ARRIVEES:], nx]); Ty = np.concatenate([Ty[ARRIVEES:], ny])
            pop.u = np.concatenate([pop.u[:, ARRIVEES:],
                                    np.repeat(pop.u.mean(1, keepdims=True), ARRIVEES, 1)], 1)
        for _ in range(PAS):
            idx = pop.tirer(NB)
            pop.pas(TX[idx], Ty[idx], idx)

        # ---- LE CLASSEMENT : la seule chose qui change d'un régime à l'autre
        if regime == "AUTO":
            idx = pop.tirer(TEST)
            note = (pop.predire(TX[idx]) == Ty[idx]).mean(1)          # chacun sur SON échantillon
        elif regime == "FIGE":
            note = (pop.predire(VX) == Vy).mean(1)
        elif regime == "FRAIS":
            FX, Fy = W.test_frais(g)
            note = (pop.predire(FX) == Fy).mean(1)
        else:  # ANCRE
            FX, Fy, iw = W.test_vise(g, c)
            note = ((pop.predire(FX) == Fy) * iw).sum(1) / iw.sum()
        o = np.argsort(note, kind="stable")
        best = int(o[-1])
        pf = pop.poids_fatale()
        tr.append(dict(g=g, note=float(note[best]), note_moy=float(note.mean()), pf=float(pf.mean()),
                       lpf=float(np.log(pf).mean()), eps=float(pop.eps.mean()), llr=float(pop.llr.mean()),
                       lwd=float(pop.lwd.mean())))

        if g in pj_points:                               # LE JUGE, que le système ne voit jamais
            JX, Jy = W.juger(g)
            rap, v30, simple, aord, fa = lecture(pop.predire(JX), Jy)
            jg.append(dict(g=g, etat=float(W.etat(g)), note_b=float(note[best]),
                           V30=float(v30.mean()), V30_b=float(v30[best]),
                           simple=float(simple.mean()), simple_b=float(simple[best]),
                           Rf=float(rap[:, FATALE].mean()), Rf_b=float(rap[best, FATALE]),
                           Aord=float(aord.mean()), Aord_b=float(aord[best]),
                           FA=float(fa.mean()), gel=float((fa > 0.5).mean())))

        # ---- LA SÉLECTION : les derniers remplacés par des copies mutées des premiers
        pop.copier(o[:NREMP], rng.choice(o[-NREMP:], NREMP, replace=False))
    fin = mesures_finales(pop, W) if finales else None
    return dict(tr=tr, juge=jg, fin=fin, recycle=W.recycle.tolist())
