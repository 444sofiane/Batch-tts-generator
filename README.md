# Générateur de TTS par lot

Génère plusieurs extraits audio et les concatène par groupe en fichiers
`.wav`. Quatre moteurs sont disponibles : le modèle TTS PyTorch de Kyutai
(`kyutai-labs/delayed-streams-modeling`, par défaut),
[Tortoise-TTS](https://huggingface.co/spaces/Manmay/tortoise-tts) (via
`--model tortoise`), [Breeze-TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
(via `--model breeze`), et l'API cloud de
[Cartesia](https://play.cartesia.ai/text-to-speech) (via `--model cartesia`).

## 1. Installer les dépendances

Nécessite Python 3.12. Si vous ne l'avez pas,
[uv](https://docs.astral.sh/uv/) peut l'installer pour vous sans toucher à
votre Python système :

```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

(Ou avec pip classique sur un Python 3.12 déjà installé :
`pip install -r requirements.txt`)

**Installez ensuite `torch` séparément**, selon votre matériel (il n'est pas
dans `requirements.txt` car la bonne version dépend de la présence d'un GPU,
et si oui de la version CUDA supportée par votre pilote) :

- **CPU uniquement** (pas de GPU, ou GPU trop ancien pour CUDA) :
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **GPU NVIDIA** : vérifiez la version CUDA maximale supportée par votre
  pilote avec `nvidia-smi`, puis choisissez une version de torch égale ou
  *inférieure* — une version plus récente que ce que supporte le pilote
  échouera à l'initialisation. Par exemple, pour un pilote supportant
  jusqu'à CUDA 12.7 :
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```

## 2. Rédiger votre fichier d'entrée

Voir `input.example.txt`. Une ligne `# nom` démarre un groupe ; les lignes
non vides qui suivent sont synthétisées dans l'ordre puis concaténées dans
`output/<nom>.wav`. Une ligne vide termine le groupe.

## 3. Lancer le script

Lancez le script sans aucun argument (`python generate_and_concat.py`) pour
une configuration guidée qui demande votre fichier d'entrée, le moteur et la
voix, puis affiche la commande équivalente avant de l'exécuter — copiez-la
pour scripter les lancements suivants (par exemple avec `nohup`). Toutes les
options `--flag` ci-dessous fonctionnent toujours exactement comme avant ;
l'assistant n'est qu'une autre façon de construire la même commande.

```bash
python generate_and_concat.py input.example.txt --output-dir output --gap-ms 300
```

Le premier lancement télécharge les poids du modèle depuis Hugging Face
(quelques Go, mis en cache ensuite). Sur CPU, la génération est lente :
comptez environ 30 à 45 secondes par phrase courte, quel que soit le
nombre de lancements — ce temps ne diminue pas avec la mise en cache. Un
GPU compatible (voir ci-dessus) est bien plus rapide via `--device cuda`.

## Options

- `--language` — `en` (par défaut) ou `fr`. Le modèle lui-même
  (`kyutai/tts-1.6b-en_fr`) est un modèle bilingue unique qui gère la
  langue du texte fourni en entrée — cette option choisit seulement une
  voix par défaut adaptée (voix anglaise ou française), elle ne change
  pas de modèle.
- `--voice` — voix à utiliser (voir `kyutai/tts-voices` sur Hugging Face
  pour les options disponibles), chemin relatif dans ce dépôt. Prend le
  dessus sur `--language`.
- `--device` — `cpu` (par défaut) ou `cuda` si vous avez installé une
  version de torch compatible avec CUDA.
- `--gap-ms` — silence inséré entre les extraits d'un même groupe (300ms
  par défaut).

## Générer avec toutes les voix

Pour constituer un jeu de données d'entraînement sur de nombreuses voix,
trois options lancent votre texte contre tout un ensemble de voix au lieu
d'une seule :

- `--all-voices` — toutes les voix de `kyutai/tts-voices` (901+ fichiers).
- `--all-fr` — uniquement les voix françaises (`cml-tts/fr/`, ~70 fichiers).
- `--all-eng` — uniquement les voix anglaises Expresso (`expresso/`, ~103
  fichiers).

Ces options ignorent `--voice`/`--language`. La sortie est plate, pas
organisée par dossier de voix : `<output-dir>/<groupe><e si la voix est une
variante "_enhanced"><index de la voix>.wav` — par exemple `intro1.wav`,
`introe2.wav`. Un fichier `voices_manifest.txt` est écrit à côté, qui
associe chaque index à sa voix source.

Comme cela peut représenter des centaines de voix x chaque extrait de votre
texte, le script affiche d'abord une estimation approximative du temps et
demande confirmation — passez `--yes` pour l'ignorer (nécessaire en
exécution non interactive, par exemple sous `nohup`). Utilisez
`--voice-limit N` pour tester sur les N premières voix avant de lancer
l'exécution complète. C'est reprenable : relancer la même commande ne
génère que les fichiers de sortie manquants, donc une exécution interrompue
ne perd pas sa progression.

```bash
nohup python generate_and_concat.py input.txt --output-dir output --all-fr --device cuda --yes > run.log 2>&1 &
```

## Utiliser le moteur Tortoise-TTS

Passez `--model tortoise` pour remplacer Kyutai par
[Tortoise-TTS](https://huggingface.co/spaces/Manmay/tortoise-tts). C'est un
modèle distinct, avec des compromis différents :

- **Anglais uniquement** — combiner `--model tortoise` avec `--language fr`
  est rejeté.
- **Les voix sont des presets intégrés**, pas un chemin dans un dépôt de
  voix HF : passez un nom de preset à `--voice` (par ex. `tom`, `angie`,
  `lj` ; voir `tortoise/voices/` dans le package installé pour la liste
  complète).
- **Bien plus lent que Kyutai**, surtout sur CPU — un GPU est fortement
  recommandé. `--tortoise-preset` contrôle le compromis qualité/vitesse :
  `ultra_fast`, `fast` (par défaut), `standard`, ou `high_quality`.
- `--all-voices` fonctionne comme pour Kyutai, mais parcourt les presets
  intégrés de Tortoise au lieu du dépôt de voix HF. `--all-fr`/`--all-eng`
  ne s'appliquent pas (les voix Tortoise ne sont pas séparées par langue)
  et sont rejetées avec `--model tortoise`.

### Fonctionnement du modèle Tortoise

`TextToSpeech()` (dans `load_tortoise_tts()`) ne permet pas de choisir un
modèle — il charge toujours le même jeu de poids pré-entraînés, depuis le
dépôt Hugging Face
[`Manmay/tortoise-tts`](https://huggingface.co/spaces/Manmay/tortoise-tts),
mis en cache sous `~/.cache/tortoise/models` après le premier lancement.
Ce n'est pas un modèle unique mais un pipeline de plusieurs réseaux :

- **`autoregressive.pth`** — le modèle principal ; transforme le texte
  d'entrée en une séquence de jetons audio, conditionnée par les
  échantillons de voix fournis.
- **`clvp2.pth`** (et éventuellement `cvvp.pth`) — évaluent plusieurs
  candidats générés et ne gardent que ceux qui correspondent le mieux au
  texte et à la voix ciblée.
- **`diffusion_decoder.pth`** — transforme les jetons audio retenus en un
  spectrogramme mel via un processus de diffusion. C'est l'étape lente ;
  `--tortoise-preset` contrôle le nombre d'étapes de diffusion effectuées
  (`ultra_fast` = le moins d'étapes/qualité la plus basse, `high_quality`
  = le plus d'étapes/le plus lent).
- **`vocoder.pth`** — convertit le spectrogramme mel en forme d'onde
  audio finale.

`--voice` et `--tortoise-preset` sont donc les deux seuls réglages que ce
script permet de changer — quels échantillons de conditionnement sont
utilisés, et combien de calcul le décodeur de diffusion y consacre. Les
poids eux-mêmes ne sont pas remplaçables sans modifier
`load_tortoise_tts()` pour lui passer un `models_dir` personnalisé.

**Installation :** Tortoise-TTS est une dépendance séparée et plus lourde,
non incluse dans `requirements.txt`, et elle épingle un `transformers`/
`tokenizers` ancien qui n'a pas de wheel précompilée pour Python 3.12 (le
`.venv` principal) — l'installer là échoue directement, ou nécessite de
compiler `tokenizers` depuis les sources (toolchain Rust, avec en plus des
conflits possibles entre versions anciennes et récentes des dépendances).
La solution la plus simple est un **venv Python 3.11 séparé** dédié à
Tortoise, puisque `tokenizers` a bien une wheel précompilée pour 3.11 :

```bash
uv venv --python 3.11 .venv-tortoise
uv pip install -r requirements.txt --python .venv-tortoise/Scripts/python.exe
uv pip install torch --index-url https://download.pytorch.org/whl/cpu --python .venv-tortoise/Scripts/python.exe
uv pip install tortoise-tts --python .venv-tortoise/Scripts/python.exe
# torchaudio n'est pas déclaré comme dépendance de tortoise-tts mais est
# requis à l'import — installez-le épinglé à la même version que le torch
# ci-dessus, sinon vous obtiendrez une erreur de chargement d'extension
# native :
uv pip install "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cpu --python .venv-tortoise/Scripts/python.exe
```

(Remplacez l'URL d'index et la version de `torch`/`torchaudio` par une
version CUDA si vous avez un GPU — voir l'étape d'installation CPU/GPU
ci-dessus — en gardant les deux versions identiques.)

```bash
.venv-tortoise/Scripts/python.exe generate_and_concat.py input.example.txt --model tortoise --voice tom --device cuda
```

## Utiliser le moteur Breeze-TTS

Passez `--model breeze` pour utiliser
[Breeze-TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2), un modèle à
poids ouverts très performant — mais une intégration plus lourde, réservée
à Linux/GPU :

- **Linux + GPU CUDA uniquement** (développé/testé sur un GPU de classe
  NVIDIA 4090 ; pas de mode CPU). Ne fonctionnera pas sur la configuration
  Windows de développement de ce projet — c'est prévu pour un serveur Linux.
- **Anglais/chinois uniquement** — `--language fr` est rejeté avec
  `--model breeze`.
- **Non installable via pip** : vous clonez le dépôt
  [breeze-tts](https://github.com/breezeblue-ai/breeze-tts) et ses packages
  `breeze_infer`/`models` sont importés directement depuis ce clone —
  pointez `--breeze-repo-dir` dessus, et `--breeze-model-dir` vers les
  poids du modèle téléchargés (un téléchargement séparé depuis Hugging
  Face).
- **Pas de catalogue de voix** — `--voice`/`--all-voices`/`--all-fr`/
  `--all-eng` sont tous rejetés avec `--model breeze`. Une voix est soit :
  - **clonée** à partir d'un extrait de référence : `--breeze-ref-audio
    ref.wav --breeze-ref-text "transcription exacte de ref.wav"`
  - **conçue** à partir d'une description textuelle, sans audio de
    référence : simplement `--breeze-instruction "une voix calme et
    grave"`
  - ou les deux ensemble (**direction de voix**) : audio/texte de
    référence *plus* une instruction, qui clone l'identité du locuteur de
    référence tout en superposant l'instruction de diction.
- `--breeze-cfg-scale` (par défaut `1.0`, comme dans `infer.py` — les
  exemples d'utilisation de Breeze utilisent `4` pour la conception/
  direction de voix) et `--breeze-seed` (par défaut `42`) ajustent la
  génération ; `--breeze-fast` active le mode rapide de Breeze
  (préchauffe/CUDA graphs).

**Installation** (sur le serveur GPU Linux, pas cette machine de
développement) :

```bash
git clone https://github.com/breezeblue-ai/breeze-tts.git
cd breeze-tts && python -m pip install -r requirements.txt && cd ..
hf download BreezeBlue/Breeze-TTS-2 --local-dir breeze-tts-2-weights
```

```bash
python generate_and_concat.py input.example.txt --model breeze \
  --breeze-repo-dir breeze-tts --breeze-model-dir breeze-tts-2-weights \
  --breeze-ref-audio ref.wav --breeze-ref-text "Transcription exacte de ref.wav"
```

**Remarque :** les poids du modèle Breeze-TTS-2 sont sous la licence
Research and Non-Commercial de BreezeBlue (le code d'inférence lui-même est
sous Apache 2.0) — un usage commercial nécessite un abonnement payant via
breezeblue.ai.

**Réserve sur les tests :** ce moteur a été implémenté à partir du code
source publié de Breeze (`infer.py`) mais n'a pas été exécuté de bout en
bout, car cela nécessite Linux + un GPU CUDA que cet environnement de
développement n'a pas. Testez-le sur votre serveur avant de vous y fier.

## Utiliser le moteur Cartesia

Passez `--model cartesia` pour utiliser les modèles Sonic de
[Cartesia](https://play.cartesia.ai/text-to-speech). Contrairement aux trois
autres moteurs, c'est une **API cloud, pas un modèle local** — pas de
téléchargement, pas de GPU, mais un compte est nécessaire et chaque extrait
appelle leurs serveurs :

- **Nécessite une variable d'environnement `CARTESIA_API_KEY`**. Récupérez
  une clé sur [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) et
  définissez-la avant de lancer le script (valable seulement pour la session
  de terminal en cours) :
  ```powershell
  # PowerShell
  $env:CARTESIA_API_KEY = "sk_car_..."
  ```
  ```bash
  # bash
  export CARTESIA_API_KEY=sk_car_...
  ```
  Ou, pour éviter de la redéfinir à chaque session, copiez `.env.example`
  vers `.env` et renseignez la clé — elle est chargée automatiquement (et
  `.env` est ignoré par git, donc jamais commité). Une variable
  d'environnement réellement exportée est toujours prioritaire sur `.env`
  si les deux sont définies.
  Son absence est vérifiée en amont et rejetée avec un message clair avant
  toute génération.
- **Les voix sont celles de votre bibliothèque Cartesia**, pas un nom de
  fichier ni un preset : passez le `voice_id` (un UUID) d'une voix à
  `--voice`. Trouvez-le en choisissant une voix sur
  [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) et en copiant
  son id, ou via `client.voices.list()` dans le SDK.
- **Anglais et français** sont tous deux supportés — `--language` est
  transmis directement à l'API de Cartesia, au lieu de seulement choisir une
  voix par défaut comme pour Kyutai. (L'API de Cartesia accepte d'autres
  codes de langue, mais l'option `--language` de ce script est pour l'instant
  limitée à `en`/`fr`, comme pour les autres moteurs.)
- `--cartesia-model` choisit le modèle (par défaut `sonic-2`, une version
  stable et figée plutôt qu'un alias mouvant `sonic-latest`, pour qu'un lot
  généré aujourd'hui sonne pareil si vous le régénérez plus tard). Voir
  [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) pour les
  autres modèles disponibles (par ex. `sonic-turbo` pour une latence plus
  faible).
- `--all-voices` parcourt toutes les voix de votre compte Cartesia au lieu
  d'un catalogue fixe. **Chaque extrait est un appel API payant** — le
  message de confirmation avant lancement met en garde sur les
  crédits/quota du compte plutôt que d'afficher une estimation de temps,
  mais aucune estimation en euros/dollars n'est calculée ; vérifiez les
  tarifs actuels sur
  [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) avant un
  lancement à grande échelle. `--all-fr`/`--all-eng` ne s'appliquent pas et
  sont rejetées avec `--model cartesia`.

**Installation :** le package `cartesia` est une dépendance séparée, non
incluse dans `requirements.txt` (un client API léger, sans dépendances ML
lourdes — s'installe proprement dans le `.venv` principal, pas besoin d'un
venv séparé) :

```bash
uv pip install cartesia
```

```powershell
# PowerShell
$env:CARTESIA_API_KEY = "sk_car_..."
python generate_and_concat.py input.example.txt --model cartesia --voice e07c00bc-4134-4eae-9ea4-1a55fb45746b
```
```bash
# bash
export CARTESIA_API_KEY=sk_car_...
python generate_and_concat.py input.example.txt --model cartesia --voice e07c00bc-4134-4eae-9ea4-1a55fb45746b
```

**Réserve sur les tests :** ce moteur a été implémenté et vérifié par
rapport au code source/exemples publiés du SDK Python de Cartesia (forme
des imports, paramètres de `tts.generate_sse`, `voices.list()`), mais n'a
pas été testé contre l'API réelle, faute d'un compte/clé API Cartesia dans
cet environnement de développement. Testez un seul extrait avant de lancer
`--all-voices`.

## Structure du projet

L'implémentation se trouve dans le paquet `tts_batch` ; `generate_and_concat.py`
à la racine du dépôt est un point d'entrée minimal (`python
generate_and_concat.py ...` fonctionne toujours exactement comme avant).

- `tts_batch/input_parsing.py` — analyse le format d'entrée `# nom` / lignes
  d'extraits.
- `tts_batch/audio.py` — concaténation, boucle de génération par voix
  reprenable, et écriture du manifeste des voix, partagées par tous les
  moteurs.
- `tts_batch/backends/` — un fichier par moteur TTS (`kyutai.py`,
  `tortoise.py`, `breeze.py`, `cartesia.py`), chacun possédant ses propres
  options CLI, sa validation d'arguments et sa logique de génération.
  `base.py` documente l'interface qu'un nouveau moteur doit implémenter.
- `tts_batch/cli.py` — assemble le parseur argparse à partir des options
  communes et de celles de chaque moteur, et délègue la validation au
  moteur sélectionné.
- `tts_batch/interactive.py` — l'assistant de configuration guidée (voir
  « Lancer le script » ci-dessus).
- `tts_batch/runner.py` — exécute une commande entièrement analysée/validée.

Ajouter un cinquième moteur consiste à créer un nouveau fichier dans
`tts_batch/backends/` implémentant la forme documentée dans `base.py`, puis
à l'ajouter au dictionnaire `BACKENDS` dans
`tts_batch/backends/__init__.py` — rien d'autre n'a besoin de changer.
