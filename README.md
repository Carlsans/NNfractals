# NNfractals

A genetic algorithm that evolves mathematical fractal formulas, judges them
with neural aesthetic models, and keeps only the beautiful ones.

## Install

**[INSTALL.md](INSTALL.md)** (English) · **[INSTALL.fr.md](INSTALL.fr.md)** (Français)

One command on Debian/Ubuntu, Fedora, or Arch — no GPU required:

```sh
git clone https://github.com/Carlsans/NNfractals.git
cd NNfractals
./scripts/install-deps.sh
```

---

## English

### What is NNfractals?

NNfractals is a Rust genetic algorithm that autonomously evolves mathematical
fractal formulas — sparse iterated complex maps built from a library of 58
holomorphic basis functions — renders them on GPU or CPU, and judges them
with an ensemble of neural aesthetic models (plus an optional personal
"taste" model trained from your own side-by-side ratings). Only fractals
that clear that bar get saved; everything else is discarded automatically.
It runs unattended, indefinitely, as a background daemon that mutates,
crosses over, restarts on stagnation, and deduplicates its own output.

Three companion GUIs make the results usable day to day:

- **Viewer** — deep-zoom, pan, and save any individual fractal, with a dozen
  colormaps.
- **Browser** — a sortable/filterable gallery of everything the evolution
  has produced, pairwise rating to train your own taste model, and a
  "Starred" folder for favorites.
- **Launcher** — the front door: start/stop/monitor evolution instances,
  train the taste model, rescore a folder, deduplicate near-identical
  fractals, and watch live CPU/GPU/RAM usage.

See **[PRINCIPLES.md](PRINCIPLES.md)** for the full architecture reference
(the 58 basis functions, the fitness function, the save gate, the genetic
operators, and more).

## Français

### Qu'est-ce que NNfractals ?

NNfractals est un algorithme génétique écrit en Rust qui fait évoluer de
façon autonome des formules fractales mathématiques — des applications
complexes itérées et éparses, construites à partir d'une bibliothèque de 58
fonctions de base holomorphes —, les affiche sur GPU ou CPU, puis les
évalue à l'aide d'un ensemble de modèles esthétiques neuronaux (avec en
option un modèle de « goût » personnel entraîné à partir de vos propres
comparaisons). Seules les fractales qui franchissent ce seuil sont
conservées ; le reste est automatiquement écarté. Le programme tourne sans
surveillance, indéfiniment, comme un démon en arrière-plan qui mute, croise,
redémarre en cas de stagnation, et dédoublonne lui-même sa production.

Trois interfaces graphiques complètent l'ensemble au quotidien :

- **Viewer** — zoom profond, déplacement et sauvegarde d'une fractale, avec
  une douzaine de palettes de couleurs.
- **Browser** — une galerie triable/filtrable de toute la production de
  l'évolution, une notation par paires pour entraîner votre propre modèle
  de goût, et un dossier « Starred » pour les favoris.
- **Launcher** — le point d'entrée : démarrer/arrêter/surveiller les
  instances d'évolution, entraîner le modèle de goût, renoter un dossier,
  dédupliquer les fractales quasi identiques, et suivre en direct
  l'utilisation CPU/GPU/RAM.

Voir **[PRINCIPLES.md](PRINCIPLES.md)** (en anglais) pour la référence
complète de l'architecture.
