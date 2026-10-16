# Consignes pour les sessions cloud — clé 39

Ce dépôt contient une expérience **pré-enregistrée** (`PREENREGISTREMENT_39.md` ; empreintes du
code, du plan et des données dans `MANIFESTE39.sha256`). Chaque session cloud exécute **une tranche**
du plan (`plan39.json`), telle qu'écrite, et pousse ses résultats sur **sa propre branche de travail**.
Rien d'autre : pas d'analyse, pas de modification, pas d'initiative.

## Si le message dit « Tranche K »

0. **Les données.** Il faut un dossier `donnees/` contenant `cifar16_a.npz`, `cifar16_b.npz` et
   `cifar16_t.npz`. Ce dossier reste local et ne doit jamais être commité. Prends-les dans le dépôt public
   de la clé 36, qui contient les mêmes fichiers :
   `git clone --depth 1 https://github.com/Lucas2a63/cle36 /tmp/cle36 && mkdir -p donnees && cat /tmp/cle36/cifar16_a.npz.part* > donnees/cifar16_a.npz && cat /tmp/cle36/cifar16_b.npz.part* > donnees/cifar16_b.npz && cp /tmp/cle36/cifar16_t.npz donnees/`
1. **Vérifier que rien n'a bougé** : `sha256sum -c MANIFESTE39.sha256`. Si une ligne échoue, arrête-toi
   et signale-le, sans rien lancer.
2. **numpy** : `python3 -c "import numpy; print(numpy.__version__)"`. S'il manque :
   `pip install numpy` (si pip refuse : `pip install --break-system-packages numpy`).
3. **Contrôle** : `python3 controle39.py` doit afficher « bon » trois fois. S'il affiche un écart,
   arrête-toi et signale l'écart, sans rien lancer.
4. **Reprise éventuelle** : `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` puis
   `python3 rassembler39.py K`. Si une session précédente avait commencé la tranche K, ses courses
   finies sont récupérées et ne seront pas refaites.
5. **Lancer en arrière-plan** :
   `nohup python3 lancer39.py --tranche K > journal_K.log 2>&1 &`
   La première ligne du journal dit combien de courses sont à faire. Une tranche dure de deux à trois
   heures sur quatre processus.
6. **Surveiller sans rendre la main.** Répète, un appel à la fois, avec un délai d'outil de 600 000 ms :
   `sleep 540; tail -n 3 journal_K.log`.
   **Ne termine pas ton tour tant que la tranche tourne** : une session inactive est arrêtée et son
   calcul perdu. Dès qu'une course est finie, puis toutes les trois vérifications environ, sauvegarde
   sur ta branche de travail :
   `git add -f resultats/tranche_K.json journal_K.log && git commit -m "tranche K : partiel" && git push -u origin HEAD`
7. **Si le calcul s'est arrêté** (`pgrep -f "lancer39.py --tranche"` ne renvoie rien) sans que le journal
   n'affiche `TRANCHE K TERMINÉE`, relance la commande de l'étape 5 : elle reprend là où elle s'était
   arrêtée, sans refaire les courses finies.
8. **À la fin** (`TRANCHE K TERMINÉE` dans le journal) : dernier commit et `git push -u origin HEAD`,
   puis réponds par les cinq dernières lignes du journal et le nom de ta branche. C'est terminé.

## Ce que tu ne fais jamais

- Modifier un fichier suivi du dépôt (code, plan, pré-enregistrement, ces consignes) — même pour
  « corriger » ou « améliorer ». Si quelque chose ne marche pas, arrête-toi et décris l'erreur.
- Analyser, résumer ou commenter les résultats : le verdict se fait ailleurs, selon la règle écrite
  avant les courses.
- Ouvrir une pull request, pousser sur `main` ou sur une autre branche que la tienne.
- Tout accès réseau autre que `pip install numpy`, `git` sur ce dépôt et, à l'étape 0, la copie du
  dépôt public de la clé 36.
- Lancer une autre tranche que celle du message, ou plus d'une tranche à la fois.
