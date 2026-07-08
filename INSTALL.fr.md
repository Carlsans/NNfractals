# Installer NNfractals

*[This guide is also available in English: INSTALL.md](INSTALL.md)*

NNfractals est un moteur d'évolution de fractales par algorithme génétique
écrit en Rust, avec trois interfaces graphiques optionnelles (viewer,
browser, launcher) et un programme Python auxiliaire pour la notation
esthétique. Ce guide permet de passer d'un clone tout neuf à une build
fonctionnelle sur une machine Linux de bureau.

**Un GPU est totalement optionnel.** Tout fonctionne sur une machine sans
GPU — voir [GPU ou CPU](#6-gpu-ou-cpu--aucun-gpu-requis) ci-dessous.

## 1. Prérequis

- Une distribution Linux de bureau — Debian/Ubuntu, Fedora, ou Arch (et
  leurs dérivées : Mint, Pop!_OS, RHEL/CentOS, Manjaro, EndeavourOS, …).
- `git`.
- Un accès Internet (pour récupérer les crates Rust, les paquets PyPI, et —
  au premier lancement — quelques modèles pré-entraînés depuis Hugging Face).

## 2. Installation rapide

```sh
git clone https://github.com/Carlsans/NNfractals.git
cd NNfractals
./scripts/install-deps.sh
```

Ce script unique :
1. Installe les bibliothèques système nécessaires aux interfaces graphiques
   (X11/Wayland/GL/Vulkan) via le gestionnaire de paquets de votre
   distribution (apt, dnf, ou pacman — détecté automatiquement).
2. Installe ou met à jour Rust via [rustup](https://rustup.rs) si la version
   fournie par votre distribution est trop ancienne (ce projet nécessite
   rustc ≥ 1.85, pour l'édition 2024 de Rust).
3. Crée un environnement virtuel Python isolé dans `.venv/` et installe la
   bonne version de `torch` — CPU uniquement ou compatible CUDA, selon
   qu'un GPU NVIDIA est détecté ou non — ainsi que toutes les autres
   dépendances Python.
4. Compile l'ensemble : `cargo build --release --features "viewer browser launcher"`.

Ce script peut être relancé sans risque : les paquets système déjà
installés, un `.venv/` existant, et une chaîne d'outils Rust déjà à jour
sont détectés et ignorés.

## 3. Installation manuelle (ou distribution non prise en charge)

Si vous n'êtes pas sous apt/dnf/pacman, ou si vous préférez le faire à la
main — voici exactement ce que le script ci-dessus automatise.

### Paquets système

| Famille de distribution | Commande |
|---|---|
| Debian/Ubuntu (apt) | `sudo apt-get install -y build-essential pkg-config curl libx11-dev libxkbcommon-dev libxkbcommon-x11-0 libwayland-dev libxrandr-dev libxi-dev libxcursor-dev libgl1-mesa-dev libegl1-mesa-dev mesa-vulkan-drivers python3 python3-venv python3-pip` |
| Fedora/RHEL (dnf) | `sudo dnf install -y gcc gcc-c++ make pkgconf-pkg-config curl libX11-devel libxkbcommon-devel libxkbcommon-x11 wayland-devel libXrandr-devel libXi-devel libXcursor-devel mesa-libGL-devel mesa-libEGL-devel mesa-vulkan-drivers python3 python3-pip` |
| Arch/Manjaro (pacman) | `sudo pacman -S --needed base-devel pkgconf curl libx11 libxkbcommon libxkbcommon-x11 wayland libxrandr libxi libxcursor mesa vulkan-icd-loader python python-pip` |

Ces paquets couvrent la pile graphique `eframe`/`winit` (X11 *et* Wayland,
plus GL/Vulkan pour le rendu — y compris le rendu logiciel de secours fourni
par Mesa lorsqu'il n'y a pas de GPU) ainsi que Python 3 et venv.

### Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version   # vérifier que la version est >= 1.85
```

### Environnement virtuel Python

```sh
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# Choisissez UNE seule de ces lignes, selon votre matériel :
.venv/bin/pip install torch                                                    # GPU NVIDIA présent
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU uniquement

.venv/bin/pip install -r requirements.txt
```

### Compilation

```sh
cargo build --release --features "viewer browser launcher"
```

## 4. Utilisation

| Commande | Ce qu'elle fait |
|---|---|
| `./run.sh` | Démarre une ou plusieurs instances d'évolution en arrière-plan (la boucle génétique principale). `./run.sh --build` recompile d'abord ; `./run.sh N` démarre N instances. |
| `./target/release/nnfractals-launcher` | Interface principale : démarrer/arrêter/suivre l'évolution, entraîner le modèle de goût, dédupliquer un dossier, ouvrir le browser/viewer. |
| `./target/release/nnfractals-browser` | Galerie : trier/filtrer les fractales, les comparer par paires, gérer un dossier « Starred ». |
| `./target/release/nnfractals-viewer <fichier.nn>` | Rendu/zoom/sauvegarde d'une fractale unique. |

Les `--features` utilisées à la compilation déterminent quels binaires
existent — `viewer`/`browser`/`launcher` sont indépendantes
(`cargo build --release --features browser`, par exemple, ne compile que
`nnfractals` et `nnfractals-browser`). Le moteur d'évolution (`nnfractals`)
lui-même ne nécessite aucune feature particulière.

## 5. Notes sur le premier lancement

Le programme auxiliaire de notation esthétique (utilisé par la fonction de
fitness de l'évolution, par les boutons « Train taste model »/« Rescore » du
launcher, et par la déduplication) télécharge plusieurs modèles
pré-entraînés depuis Hugging Face **lors de leur première utilisation** :
SigLIP, DINOv2, les poids NIMA/TOPIQ/MUSIQ de `pyiqa`, et Aesthetic
Predictor v2.5. À prévoir :
- Un accès Internet nécessaire une fois par modèle (mis en cache ensuite
  dans `~/.cache/huggingface`).
- Quelques Go d'espace disque.
- Le premier score après le démarrage sera lent le temps que les modèles se
  chargent ; les suivants seront rapides.

## 6. GPU ou CPU — aucun GPU requis

Un GPU est optionnel partout dans ce projet :

- **Évolution et rendu** (`nnfractals`, `nnfractals-viewer`) sont toujours
  compilés avec le support de rendu GPU (`wgpu-backend`), mais au démarrage
  ils recherchent un GPU utilisable et, si aucun n'est trouvé, affichent
  `[gpu] No wgpu adapter — using CPU renderer.` puis basculent
  automatiquement sur un moteur de rendu CPU parallèle (Rayon). Même
  binaire, même commande — juste plus lent sans GPU.
- **Notation esthétique** (le programme Python auxiliaire, l'entraînement,
  la déduplication) vérifie `torch.cuda.is_available()` avant chaque appel
  de modèle et bascule automatiquement sur le CPU si besoin.
  `install-deps.sh` installe la version CPU uniquement de `torch` lorsqu'aucun
  GPU NVIDIA n'est détecté, pour éviter de télécharger plusieurs Go de CUDA
  inutilisables.
- **Les interfaces Browser et Launcher** ne touchent pas du tout au chemin
  de calcul GPU — elles utilisent OpenGL pour le rendu, que Mesa fournit en
  version logicielle (`llvmpipe`) en l'absence de GPU.

## 7. Intégration au bureau (optionnel)

```sh
./target/release/nnfractals-launcher --install-desktop   # ajoute NNFractals au menu des applications
bash dist/install-viewer.sh                               # + ouvre les fichiers .nn dans le viewer par double-clic
```

## 8. Dépannage

- **`error: package requires... edition2024` / la compilation échoue avec
  un vieux rustc** — la version de Rust de votre distribution est trop
  ancienne. Lancez
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` puis
  `source "$HOME/.cargo/env"`, et recompilez. Ce projet nécessite rustc ≥ 1.85.
- **La fenêtre graphique ne s'ouvre pas / erreurs X11 ou Wayland** —
  vérifiez que la liste de paquets système du §3 a bien été installée en
  entier pour votre distribution ; en session Wayland en particulier,
  `libwayland-dev`/`wayland-devel`/`wayland` (selon la distribution) doit
  être présent.
- **`python3-venv` / « ensurepip is not available »** — sur Debian/Ubuntu,
  `venv` est fourni par le paquet séparé `python3-venv` ; installez-le puis
  relancez `python3 -m venv .venv`.
- **La notation esthétique ne produit silencieusement aucun score** —
  consultez `aesthetic_sidecar.log` (évolution) ou `train_pref.log` (tâches
  du launcher) à la racine du projet ; ces fichiers capturent la
  sortie d'erreur/les tracebacks du programme Python auxiliaire.
- **Vous remarquez une feature Cargo `ndarray-backend` dans `Cargo.toml`** —
  ignorez-la, elle n'a actuellement aucun effet (code mort). Le repli sur
  CPU décrit au §6 fonctionne déjà via la build par défaut avec
  `wgpu-backend` ; il n'est jamais nécessaire de passer
  `--features ndarray-backend`.
