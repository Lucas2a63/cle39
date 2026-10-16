# Clé 39 — Le système entier contre la recette standard, à calcul égal

Pré-enregistrement, écrit le 16 octobre 2026 avant toute course du plan. Projet Singularité (Lucas).

## La question

Trente-huit clés ont donné trente-deux règles de conception pour qu'un système qui apprend reste vrai et
protège ce qui est fatal : un juge extérieur et frais, visé sur le fatal ; peser juste (×30) ; laisser
apprendre avant de peser (brièvement, et selon l'architecture) ; croire la population plutôt que le
champion. Aucune ne dit si ce système **apprend mieux** qu'on ne le ferait autrement. C'est la question de
cette clé : à calcul d'entraînement égal et à données égales, le système entier bat-il la recette
standard — un réseau seul, bien pesé, qui reçoit tout le calcul ? Si oui, nos règles deviennent une
architecture. Si non, elles servent la sûreté et l'honnêteté, pas la capacité.

## Le montage

**Le monde, les données, l'ancre et le juge** sont ceux des clés 36 à 38, inchangés : CIFAR-10 réduit en
16×16 ; le chat est fatal (3 % du monde, ×30 pour le juge) ; le monde envoie 500 images neuves par
génération dans un tampon de 5 000 ; l'ancre tire à chaque génération 200 images neuves de la réserve, dont
40 chats, pesées au coût ; le juge lit les 10 000 images de test, que rien ni personne n'apprend ; 120
générations ; monde fixe. À graine égale, tous les bras reçoivent exactement les mêmes images du flux et
les mêmes ancres.

**L'unité de calcul** est le pas : un pas de gradient d'un réseau sur un lot de 64 images. Le système
entier dépense 32 × 100 = 3 200 pas par génération (384 000 en tout). Chaque bras en dépense autant, sauf
COURT. L'évaluation par l'ancre (200 images par réseau et par génération, sans gradient) représente moins
de 1 % du calcul ; le juge est un instrument de mesure, hors du compte.

**Les bras** (convolutif : 16 puis 32 filtres ; perceptron : 768 → 128 → 10) :

| Bras | Ce que c'est | Pas par génération |
|---|---|---|
| **SYS** | Le système entier : 32 réseaux ; l'ancre les note ; à chaque génération, le quart le moins bien noté est remplacé par des copies mutées du quart le mieux noté (poids appris copiés ; génome muté : taux d'apprentissage, décroissance, curation) ; ×30 dans la perte à la génération 10 pour le convolutif, dès 0 pour le perceptron (contrainte 31). Livre le réseau le mieux noté par l'ancre à la dernière génération. Aux graines 0 à 11, ce sont les courses publiées des clés 36 (perceptron) et 37–38 (convolutif), refaites au bit près avec des mesures en plus. | 32 × 100 |
| **INDEP** | La même population sans sélection : 32 réseaux indépendants aux génomes tirés au hasard (une recherche aléatoire d'hyperparamètres), même bascule ; livre aussi le mieux noté. | 32 × 100 |
| **SEUL** | La recette standard : un réseau seul de la même architecture ; ×30 dès le départ ; tirage uniforme dans le tampon ; taux d'apprentissage en cosinus, de lr₀ à 0 sur toute la course ; décroissance des poids 1e-4. Tout le calcul de la population dans un seul réseau. | 3 200 |
| **SEUL10** | SEUL, avec la bascule à la génération 10 (convolutif seulement) : la contrainte 31 appliquée à la recette standard. | 3 200 |
| **GRAND** | La recette standard avec un réseau 32 fois plus coûteux par image : convolutif à 96 et 186 filtres (31,98 fois le calcul par image), perceptron à 4 096 neurones (32,0 fois). | 100 |
| **COURT** | SEUL avec 100 pas par génération : le calcul d'un seul réseau de la population. | 100 |

**Le réglage de la recette standard.** Un praticien règle le taux d'apprentissage de son réseau sur des
données de validation. Avant ce texte, le taux de départ lr₀ de SEUL, COURT et GRAND a été choisi par un
balayage sur la graine 99, qui n'est pas dans le plan, avec l'ancre pour seule note — la moyenne de ses
notes aux dix dernières générations (2 000 images) —, jamais le juge (`balayage39.py`). Grille de départ :
lr₀ = 0,0632 × {¼, ½, 1, 2, 4} pour SEUL et COURT, × {1/32, 1/16, 1/8, ¼, ½, 1} pour GRAND ; SEUL10 prend le
taux de SEUL. Plusieurs meilleurs taux tombant au bord de cette grille, j'ai ajouté, avant tout
pré-enregistrement, une règle de prolongement — diviser ou multiplier par deux du côté du bord tant que le
meilleur y reste, trois fois au plus — et l'ai appliquée à tous les bras. Le script n'a affiché que les taux
retenus et la durée des courses ; les notes du balayage sont dans `balayage39.json`, que je ne lirai
qu'après le verdict. Grilles finales et taux retenus (multiples de 0,0632) :

- SEUL et SEUL10, convolutif : 1/32 (lr₀ = 0,0020 ; grille {1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4})
- COURT, convolutif : 1 (lr₀ = 0,0632 ; grille {1/4, 1/2, 1, 2, 4})
- GRAND, convolutif : 1 (lr₀ = 0,0632 ; grille {1/32, 1/16, 1/8, 1/4, 1/2, 1, 2})
- SEUL, perceptron : 1/32 (lr₀ = 0,0020 ; grille {1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4})
- COURT, perceptron : 1/8 (lr₀ = 0,0079 ; grille {1/16, 1/8, 1/4, 1/2, 1, 2, 4})
- GRAND, perceptron : 1/16 (lr₀ = 0,0040 ; grille {1/32, 1/16, 1/8, 1/4, 1/2, 1})

Pour SEUL, sur les deux architectures, le meilleur taux est encore au bord inférieur après les trois
prolongements permis : son optimum est peut-être plus bas — une limite de ce réglage, qui pourrait
désavantager SEUL. Le système entier n'est pas réglé : sa sélection règle elle-même son génome, dans les bornes fixées depuis la clé 36. Le
balayage a coûté 5,6 heures de calcul, hors du compte de la course.

**Les graines** : 0 à 11 pour le convolutif (72 courses) ; 0 à 23 pour le perceptron (120 courses), plus
bruyant et moins coûteux. 192 courses, environ 51 heures de calcul.

**Les mesures.** Le *réseau livré* est le mieux noté par l'ancre pour SYS et INDEP, le réseau lui-même pour
les autres bras.

- **V_fin** : la vérité ×30 du juge pour le réseau livré, à la génération 119.
- **V_vie** : la même, en moyenne sur les 16 passages du juge 0, 8, 16, …, 112, 119 (le réseau que le
  système livrerait à ce moment-là).
- **H** (honnêteté) : la note que l'ancre donne au réseau livré, moins sa vérité selon le juge, en moyenne sur
  les 11 passages 40, 48, …, 112, 119. Pour un réseau seul, la note de l'ancre est pratiquement sans biais ; pour le
  mieux noté de 32, elle est gonflée par la chance (la malédiction du vainqueur de la clé 36).
- **E_fin** : la vérité ×30 de l'ensemble des 32 réseaux (moyenne de leurs probabilités), génération 119.
- Mesures finales (banc de stress de la clé 36, sonde, dimension effective) pour le réseau livré et pour
  l'ensemble : rapportées, sans prédiction.

## Les prédictions

Mon raisonnement : dans ce monde, les données arrivent au compte-gouttes (500 images par génération, un
tampon de 5 000). Un petit réseau seul qui reçoit 32 fois plus de pas repasse chaque image du tampon environ
410 fois au lieu de 13 : il sur-apprend son tampon et perd de sa plasticité. La population, elle, convertit
son calcul en recherche — elle découvre seule un recuit du taux d'apprentissage (de 0,15 à 0,006 dans les
clés 36–38) — et un réseau plus grand le convertit en capacité, qui sert à chaque image. Je parie donc que
la population bat le petit réseau entraîné longtemps, mais perd contre le réseau plus large.

Ces prédictions ont été écrites pendant le balayage ; je ne les ai pas changées après avoir vu les taux
retenus (seule information que le balayage m'ait montrée).

**Principales** (corrigées ensemble, voir la règle) :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **P1** | Convolutif : V_fin(SYS) > V_fin(SEUL) | +0,5 à +2 points | 60 % |
| **P2** | Perceptron : V_fin(SYS) > V_fin(SEUL) | +1 à +4 points | 70 % |
| **P3** | Convolutif : V_fin(GRAND) > V_fin(SYS) | +2 à +5 points | 75 % |
| **P4** | Perceptron : V_fin(GRAND) > V_fin(SYS) | +1 à +3 points | 65 % |

**Secondaires** :

| | Prédiction | Ampleur attendue | Confiance |
|---|---|---|---|
| **S1** | Convolutif : V_fin(SYS) > V_fin(INDEP) — la sélection bat la recherche aléatoire | +1 à +3 | 80 % |
| **S2** | Perceptron : idem | +1 à +3 | 80 % |
| **S3** | Convolutif : V_fin(SYS) > V_fin(COURT) — la population vaut ses 32 réseaux | +0,3 à +1,5 | 55 % |
| **S4** | Perceptron : idem | +0,5 à +2 | 55 % |
| **S5** | Convolutif : V_fin(SEUL10) > V_fin(SEUL) — la contrainte 31 améliore la recette standard | +0,5 à +2 | 55 % |
| **S6** | Convolutif : V_fin(SEUL) > V_fin(COURT) — le petit convolutif profite encore de 32 fois plus de pas | +0,5 à +2 | 55 % |
| **S7** | Perceptron : V_fin(COURT) > V_fin(SEUL) — 32 fois plus de pas nuisent au perceptron seul | +0,5 à +3 | 55 % |
| **S8** | Convolutif : V_vie(SEUL) > V_vie(SYS) — la recette standard protège mieux pendant l'apprentissage | +2 à +5 | 85 % |
| **S9** | Perceptron : idem | +1 à +4 | 70 % |
| **S10** | Convolutif : H(SYS) > H(SEUL) — le système se surestime, le réseau seul non | +1 à +3 | 85 % |
| **S11** | Perceptron : idem | +0,5 à +3 | 60 % |
| **S12** | Convolutif : E_fin(INDEP) > V_fin(INDEP) — l'ensemble de 32 réseaux divers bat le meilleur d'entre eux | +1 à +4 | 85 % |
| **S13** | Convolutif : \|E_fin(SYS) − V_fin(SYS)\| < 1 point — les 32 réseaux du système descendent d'un seul ancêtre : leur ensemble n'ajoute rien | — | 75 % |
| **S14** | Perceptron : E_fin(INDEP) > V_fin(INDEP) | +1 à +4 | 85 % |

**Comparaisons rapportées sans prédiction** (donc sans verdict) : l'ensemble des 32 indépendants contre
le système ; GRAND contre SEUL ; la meilleure des recettes standard contre le système ; le banc de stress,
la sonde, la dimension effective ; le rappel du chat, la précision ordinaire et les fausses alertes ; le
taux d'apprentissage découvert par la population contre le cosinus ; la durée des courses.

**Ce que chaque issue voudra dire.** Si P1–P2 et P3–P4 sont confirmées : à calcul égal, la population tire
plus du calcul qu'un petit réseau entraîné longtemps, mais moins qu'un réseau plus grand — le levier est la
taille ; nos règles servent la sûreté et l'honnêteté. Si le système bat aussi GRAND (P3–P4 démenties), nos
règles sont, à cette échelle, un gain de capacité, et il faudra monter en échelle pour voir s'il tient. Si
SEUL bat le système (P1–P2 démenties), la population gaspille son calcul : même un petit réseau seul, bien
réglé, fait mieux.

## La règle de décision

Toutes les comparaisons sont appariées par graine (mêmes images, mêmes ancres). Pour chaque test, l'écart
par graine d (A − B), le t de Student et la valeur p unilatérale exacte à n − 1 degrés de liberté (n = 12
pour le convolutif, 24 pour le perceptron).

- **Tests principaux (P1–P4)** : la méthode de Holm, au niveau unilatéral 0,025 pour l'ensemble des quatre.
  *Confirmée* si Holm la retient ; *démentie* si le p dans le sens contraire est sous 0,025 / 4 ;
  *indécidable* sinon.
- **Tests secondaires (S1–S12, S14)** : *confirmée* si p < 0,025 ; *démentie* si le p contraire < 0,025 ;
  *indécidable* sinon. Pas de correction pour leur nombre ; les p ajustés par Holm sur les 18 tests sont
  rapportés pour information.
- **Équivalence (S13)** : deux tests unilatéraux, bornes ±0,01 (un point), chacun à 0,025. *Confirmée* si
  les deux rejettent ; *démentie* si l'écart dépasse significativement une borne (p < 0,025) ; *indécidable*
  sinon.
- Plus de deux graines manquantes pour un bras : les tests qui l'utilisent sont *incomplets*.
- Pour la continuité avec les clés précédentes, le verdict selon l'ancienne règle (t > 2) est rapporté à côté.
  Elle ne décide pas : la clé 38 a montré qu'avec 12 graines, t > 2 ne vaut que p < 0,035, et qu'aucune
  correction n'était faite pour le nombre de tests.

Le script `verdict39.py`, dont l'empreinte est dans le manifeste, applique cette règle telle quelle.

## Ce qui a tourné avant ce texte

- `identite39.py` : SYS refait au bit près les trois premières générations de trois courses publiées
  (convolutif, bascule 10, graine 0 de la clé 37 et graine 4 de la clé 38 ; perceptron, ×30, graine 0 de la
  clé 36) : 57 valeurs chacune, écart 0. Un pas de SEUL est exactement le pas d'un membre de la
  population (écart 0).
- `pilote39.py` : une ou deux générations de chaque bras sur la graine 99, pour la mécanique et la durée ;
  seules les durées ont été affichées.
- `balayage39.py` : le réglage de la recette standard, décrit plus haut (41 courses) ; seuls les taux retenus
  et les durées ont été affichés.
- `controle39.py` : le contrôle de l'environnement des sessions (trois débuts de course, « bon » trois fois).

## Où et comment

Les 192 courses sont réparties en 8 tranches (`plan39.json`) : 1 à 4, les populations du convolutif ;
5 et 6, ses réseaux seuls ; 7 et 8, le perceptron. Elles tournent dans ce bac à sable et, si Lucas les
lance, dans des sessions cloud qui exécutent chacune une tranche telle qu'écrite (`CLAUDE.md`). Rien n'agit
hors du bac à sable : pas d'agent autonome, pas d'accès réseau autre que numpy, git et les données de la
clé 36.

## Les limites connues d'avance

- Un seul monde (fixe), un seul niveau de rareté du fatal, de petits réseaux en numpy, sans augmentation de
  données, en descente de gradient simple (sans moment ni Adam).
- Le calcul est égal à l'entraînement, pas à l'usage : l'ensemble coûte 32 fois plus à interroger qu'un
  réseau ; GRAND, 32 fois plus.
- La recette standard est réglée sur une seule graine, sur une grille de cinq à huit taux (prolongée au bord),
  dont le bord inférieur reste le meilleur pour SEUL ; le système entier hérite de bornes de génome fixées
  depuis la clé 36.
- La durée réelle des courses n'est pas le calcul : la population est écrite en numpy pour 32 réseaux à la
  fois, et un réseau seul tourne plus vite à calcul égal.
