# Clé 39 — Le système entier contre la recette standard, à calcul égal

Projet Singularité. Trente-huit clés ont donné des règles pour qu'un système qui apprend reste vrai et
protège ce qui est fatal : une population de réseaux, classée par un juge extérieur et frais visé sur la
classe fatale, qui pèse juste et laisse apprendre avant de peser. Cette clé demande si ce **système entier**
apprend mieux que la **recette standard** — un réseau seul, bien pesé et réglé, qui reçoit tout le calcul —
à calcul d'entraînement égal et à données égales.

Le monde : CIFAR-10 réduit en 16×16 ; la classe fatale est le **chat** (3 % du monde, ×30 pour le juge).
Six bras : le système entier (SYS), la même population sans sélection (INDEP), un réseau seul avec tout le
calcul (SEUL, et SEUL10 avec la bascule), un réseau 32 fois plus large (GRAND), un réseau seul avec le
calcul d'un seul membre de la population (COURT). Convolutif sur 12 graines, perceptron sur 24.

Les prédictions et la règle de décision ont été écrites **avant toute course** :
`PREENREGISTREMENT_39.md`. Son empreinte et celles du code, du plan et des données sont dans
`MANIFESTE39.sha256`.

## Lancer

Rien que Python 3 et numpy. Les données (`donnees/`) sont celles de la clé 36
(https://github.com/Lucas2a63/cle36) : voir l'étape 0 de `CLAUDE.md`.

```sh
sha256sum -c MANIFESTE39.sha256          # rien n'a bougé depuis le pré-enregistrement
python3 controle39.py                    # l'environnement calcule comme le bac à sable
python3 lancer39.py --tranche 1          # une tranche du plan (4 processus par défaut)
```

Le plan (`plan39.json`, écrit par `plan39.py`) compte 192 courses en 8 tranches : 1 à 4, les populations
du réseau convolutif ; 5 et 6, ses réseaux seuls ; 7 et 8, le perceptron. Chaque tranche peut tourner dans
sa propre session cloud : les consignes des sessions sont dans `CLAUDE.md`.

## Fichiers

- `echelle.py` (clé 36) et `echelle37.py` (clé 37), inchangés — le monde, les réseaux, la population,
  l'ancre, le juge, les mesures.
- `echelle39.py` — les six bras.
- `plan39.py`, `plan39.json`, `balayage39_choix.json` — les courses, leur répartition, et les taux de départ
  de la recette standard choisis par `balayage39.py` (sur une graine hors plan, notés par l'ancre seule).
- `lancer39.py` — exécute une tranche, reprend là où elle s'était arrêtée.
- `rassembler39.py` — réunit les résultats poussés sur toutes les branches.
- `controle39.py`, `controle39_attendu.json` — vérifie que l'environnement refait au bit près trois débuts
  de course.
- `identite39.py` — vérifie que le système entier refait au bit près les courses publiées des clés 36 à 38.
- `verdict39.py` — applique la règle de décision, telle qu'écrite.

## Données

CIFAR-10 : Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
https://www.cs.toronto.edu/~kriz/cifar.html — réduit en 16×16 par moyenne des blocs 2×2.
