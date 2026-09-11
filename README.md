# Générateur de TTS par lot

Génère plusieurs extraits audio et les concatène par groupe en fichiers
`.wav`. Sept moteurs sont disponibles : le modèle TTS PyTorch de Kyutai
(`kyutai-labs/delayed-streams-modeling`, par défaut),
[Tortoise-TTS](https://huggingface.co/spaces/Manmay/tortoise-tts) (via
`--model tortoise`), [Breeze-TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
(via `--model breeze`), [Piper](https://github.com/OHF-Voice/piper1-gpl) (via
`--model piper`), [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (via
`--model kokoro`), [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) (via
`--model xtts`), et l'API cloud de
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
  pas de modèle. Les deux voix par défaut (`unmute-prod-website/default_voice.wav`
  pour l'anglais, `cml-tts/fr/10087_11650_000028-0002.wav` pour le français)
  sont commerciale-safe — voir [KYUTAI_VOICE_LICENSES.md](KYUTAI_VOICE_LICENSES.md).
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
- `--all-eng` — uniquement les voix anglaises (`expresso/`, `vctk/` et
  `ears/`, ~362 fichiers).
- `--kyutai-commercial-safe` — combinable avec les trois options
  ci-dessus, ne garde que les voix dont la licence autorise explicitement
  un usage commercial (CC0/CC-BY par jeu de données) — voir
  [KYUTAI_VOICE_LICENSES.md](KYUTAI_VOICE_LICENSES.md) (**pas un avis
  juridique**, à vérifier vous-même). Exclut notamment `expresso/` et
  `ears/`, qui sont sous licence CC BY-NC (non-commerciale). Avec une
  seule `--voice`, rejette la commande en amont si la voix n'est pas sur
  cette liste.

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

## Utiliser le moteur Piper

Passez `--model piper` pour utiliser [Piper](https://github.com/OHF-Voice/piper1-gpl),
un modèle local léger basé sur `onnxruntime` (pas `torch`) : pas de GPU
nécessaire pour tourner bien plus vite que temps réel, ce qui en fait le
moteur le plus adapté de ce script pour générer de très gros volumes sans
carte graphique. En échange, la qualité audio est plus "robotique" que
Kyutai/Tortoise/Breeze.

- **De très nombreuses langues** — le catalogue de voix Piper couvre l'anglais,
  le français et des dizaines d'autres langues, mais l'option `--language` de
  ce script reste limitée à `en`/`fr` comme pour les autres moteurs (elle
  choisit seulement une voix par défaut). Passez n'importe quel id de voix à
  `--voice` pour utiliser une autre langue du catalogue.
- **Les voix sont des ids Piper** (par ex. `en_US-arctic-medium`,
  `fr_FR-siwis-medium`), au format `<langue>_<région>-<nom>-<qualité>` —
  parcourez [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples)
  pour écouter et choisir. Les fichiers de voix (`.onnx`/`.onnx.json`) sont
  téléchargés automatiquement depuis Hugging Face au premier lancement puis
  mis en cache, comme pour Kyutai.
- `--piper-length-scale` (par défaut `1.0`) ajuste la vitesse de parole :
  supérieur à `1` ralentit, inférieur à `1` accélère.
- `--all-voices` fonctionne comme pour Kyutai, mais parcourt tout le
  catalogue Piper ; `--all-fr`/`--all-eng` filtrent le catalogue par langue
  au lieu d'être rejetées.
- `--device cuda` fonctionne aussi avec Piper (via `onnxruntime-gpu`, à
  installer séparément), mais l'intérêt principal de ce moteur est justement
  de tourner vite sur CPU seul.

### Filtrer sur les voix utilisables commercialement

Les voix du catalogue Piper viennent de jeux de données aux licences très
variées (certaines interdisent explicitement l'usage commercial, d'autres
sont sous copyleft GPL/AGPL, d'autres encore n'ont pas de licence claire).
[PIPER_VOICE_LICENSES.md](PIPER_VOICE_LICENSES.md) documente ce qui a été
vérifié pour les 176 voix du catalogue actuel — **ce n'est pas un avis
juridique**, vérifiez vous-même avant de vous y fier pour un usage commercial
réel.

Passez `--piper-commercial-safe` pour que `--all-voices`/`--all-fr`/
`--all-eng` ne génèrent que les voix dont la licence permet explicitement un
usage commercial (CC0/domaine public/CC-BY/CC-BY-SA/Apache, ou vérifiées
individuellement) — les voix non-commerciales, sous copyleft, ou à la
licence incertaine sont exclues. Avec une seule `--voice`, cette option
rejette la commande en amont si la voix choisie n'est pas sur cette liste,
plutôt que de générer quand même :

```bash
python generate_and_concat.py input.example.txt --model piper \
  --all-fr --piper-commercial-safe --yes
```

Les voix par défaut du script (`en_US-arctic-medium`, `fr_FR-siwis-medium`)
sont déjà sur la liste commerciale-safe, donc un lancement sans `--voice` ni
`--all-voices` est déjà sûr de ce point de vue, avec ou sans ce flag.

### Traduire le corpus pour chaque langue Piper

Les voix Piper sont généralement monolingues. Pour éviter de faire lire du
français par une voix anglaise ou japonaise, activez la traduction avant la
synthèse :

```bash
uv pip install argostranslate deep-translator
python generate_and_concat.py input.example.txt --model piper \
  --all-voices --piper-translate --yes
```

`--piper-translate` essaie deux moteurs, dans cet ordre :

1. **[Argos Translate](https://github.com/argosopentech/argos-translate)**
   — une bibliothèque de traduction neuronale **entièrement hors-ligne**
   (CTranslate2), sans clé d'API, sans quota. Chaque paire de langues est un
   modèle téléchargé une seule fois (~60-70 Mo, mis en cache
   sous `~/.local/share/argos-translate`, en passant par l'anglais comme
   langue pivot si besoin : `fr` → `en` → `ar`). Couvre 39 des 52 familles de
   langues du catalogue Piper. Entièrement optionnel : si le paquet n'est pas
   installé, ce moteur est simplement ignoré.
2. **MyMemory**, via `deep-translator` — une vraie API cloud, utilisée
   uniquement en repli pour les langues qu'Argos ne couvre pas (environ 13,
   par ex. gallois, géorgien, télougou). Pas Google Translate :
   `GoogleTranslator` s'appuie sur du scraping d'une page HTML dont Google a
   changé la structure, ce qui le casse entièrement — toute traduction
   échoue avec `TranslationNotFound()`, quelle que soit la langue, comme
   vérifié dans cet environnement de développement. Les codes de langue
   bruts utilisés par ce script (`fr`, `en`, `ar`...) sont résolus
   automatiquement vers le format attendu par MyMemory (`fr-FR`, `en-GB`,
   `ar-EG`...).

Installer seulement l'un des deux fonctionne aussi (l'autre est simplement
ignoré) ; installer les deux donne la meilleure couverture avec le moins de
dépendance au quota MyMemory.

- **Quota gratuit MyMemory : ~5000 caractères/jour** (anonyme), extensible à
  ~50000/jour en fournissant un email de contact via
  `--piper-translation-email vous@exemple.com` — voir
  [mymemory.translated.net](https://mymemory.translated.net). Sans Argos
  installé, un run `--all-voices` (176 voix / 52 langues sur le catalogue
  Piper actuel) peut dépasser ce quota en un lancement ; avec Argos, seule
  la petite poignée de langues non couvertes par Argos y contribue.
- **`--piper-translate-only`** : ne fait que remplir le cache de traduction
  pour la/les langue(s) nécessaire(s) (téléchargeant au passage les modèles
  Argos manquants), sans charger Piper ni générer le moindre audio. Utile
  pour séparer l'étape réseau/téléchargement (plus lente, sujette au quota
  MyMemory pour les langues non couvertes par Argos) de l'étape de
  génération (rapide, locale), et pour relancer juste les traductions
  manquantes sans reprendre tout le run.
- **`--piper-translate-cached-only`** : avec `--all-voices`/`--all-fr`/
  `--all-eng`, ne génère que les voix dont la langue a déjà **toutes** ses
  phrases en cache — les autres sont ignorées plutôt que de tenter une
  traduction en direct qui pourrait échouer (quota MyMemory épuisé, par
  exemple). Pratique pour générer tout de suite ce qui est déjà traduit et
  rattraper le reste plus tard (relancez sans ce flag une fois le cache plus
  complet — voir `--piper-translate-only` ci-dessus pour le remplir sans
  générer d'audio entretemps).
- La traduction est mise en cache dans `output/piper_translations.json` (une
  entrée par triplet langue source/langue cible/phrase, quel que soit le
  moteur qui l'a produite) et chaque phrase n'est traduite qu'une fois par
  langue, y compris entre plusieurs lancements interrompus. Avant de lancer
  la génération, `--all-voices`/`--all-fr`/`--all-eng` pré-remplissent
  automatiquement le cache pour toutes les langues nécessaires (une seule
  fois par langue, pas par voix) ; les échecs sont listés avec le texte
  concerné plutôt que de faire sauter silencieusement toute une voix
  pendant la génération.
- Le français est la langue source par défaut ; utilisez
  `--piper-translation-source en` pour un corpus anglais. Testez quelques
  voix avec `--voice-limit N` avant un lancement complet : la couverture et
  la qualité de traduction dépendent des services utilisés.

**Installation :** contrairement à Tortoise/Breeze, `piper-tts` est une
dépendance légère (pas de gros stack ML) qui s'installe proprement dans le
`.venv` principal, sans venv séparé :

```bash
uv pip install piper-tts
```

```bash
python generate_and_concat.py input.example.txt --model piper --voice fr_FR-siwis-medium
```

**Réserve sur les tests :** ce moteur a été implémenté à partir de la
documentation publiée de l'API Python de Piper (`PiperVoice.load`,
`synthesize`, `SynthesisConfig`) et du format de `voices.json`, mais n'a pas
été exécuté de bout en bout dans cet environnement de développement (pas
d'installation testée de `piper-tts` ici). Testez un seul extrait avant de
lancer `--all-voices`.

## Utiliser le moteur Kokoro

Passez `--model kokoro` pour utiliser [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
un petit modèle (82M paramètres) rapide et léger sur CPU — même classe de
vitesse que Piper (moins d'une seconde par extrait court sur CPU, pas de
GPU nécessaire), avec un rendu sensiblement plus naturel :

- **8 langues, 54 voix** : anglais (américain/britannique), français,
  japonais, mandarin, espagnol, hindi, italien, portugais brésilien.
  `--language` reste limité à `en`/`fr` comme pour les autres moteurs
  (choisit seulement une voix par défaut) ; passez n'importe quel code de
  voix à `--voice` pour une autre langue du catalogue.
- **Les voix sont des codes Kokoro** (par ex. `af_heart`, `ff_siwis`), au
  format `<langue><genre>_<nom>` — la première lettre indique la langue
  (`a`/`b` anglais américain/britannique, `f` français, `j` japonais, `z`
  mandarin, `e` espagnol, `h` hindi, `i` italien, `p` portugais brésilien).
  Voir [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
  sur le dépôt du modèle pour la liste complète avec notes de qualité par voix.
- `--kokoro-speed` (par défaut `1.0`) ajuste la vitesse de parole :
  supérieur à `1` accélère, inférieur à `1` ralentit.
- `--all-voices` parcourt les 54 voix ; `--all-fr` ne contient qu'une seule
  voix (`ff_siwis`, la seule voix française du catalogue) ; `--all-eng`
  parcourt l'anglais américain et britannique (28 voix).

**Licence — toutes les voix sont utilisables commercialement**, contrairement
à Piper/Kyutai : le dépôt est sous licence Apache 2.0 dans son ensemble, et 5
voix sur 54 (4 japonaises, 1 française : `ff_siwis`, la voix française par
défaut) sont individuellement sous CC BY (jeux de données Koniwa et SIWIS) —
usage commercial autorisé également, simple obligation d'attribution.
Aucune voix du catalogue n'est non-commerciale, donc pas de flag
`--kokoro-commercial-safe` (contrairement à `--piper-commercial-safe`/
`--kyutai-commercial-safe`) : rien à exclure.

**Installation :** dépendance légère (pas de gros stack ML au-delà de
`torch`, déjà installé à l'étape 1), s'installe proprement dans le `.venv`
principal :

```bash
uv pip install kokoro
```

```bash
python generate_and_concat.py input.example.txt --model kokoro --voice ff_siwis
```

**Testé de bout en bout** dans cet environnement de développement :
génération réussie en anglais et en français (voix par défaut), et un
`--all-eng --voice-limit 3` confirmant la boucle multi-voix et le
partage du modèle chargé entre langues (le modèle n'est chargé qu'une
fois, seul le pipeline de phonémisation change par langue).

## Utiliser le moteur XTTS-v2

Passez `--model xtts` pour utiliser [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2),
un modèle multilingue à poids ouverts, via le fork communautaire
[idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) qui maintient le
package `coqui-tts` (l'ancien package `TTS` de Coqui AI est à l'arrêt depuis
la fermeture de l'entreprise, mais ce fork le garde compatible avec les
versions récentes de Python/torch) :

- **17 langues** dont l'anglais et le français, mais comme pour les autres
  moteurs, `--language` de ce script reste limité à `en`/`fr` — passé
  directement à XTTS comme paramètre de langue.
- **Deux façons de choisir une voix**, mutuellement exclusives :
  - **Une voix intégrée** (~58 "studio speakers" fournis avec le modèle) :
    `--voice "Claribel Dervla"` (nom exact requis — listez-les avec la
    commande ci-dessous).
  - **Un clonage à partir d'un extrait de référence** : `--xtts-speaker-wav
    ref.wav` (quelques secondes de parole propre suffisent, pas besoin de
    transcription contrairement à Breeze).
- `--all-voices` parcourt les ~58 voix intégrées (pas de `--all-fr`/
  `--all-eng`, qui sont rejetées avec `--model xtts` — utilisez `--voice-limit`
  pour tester sur un sous-ensemble).
- Lister les voix intégrées disponibles :
  ```bash
  .venv/bin/python -c "from TTS.api import TTS; print(TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False).speakers)"
  ```
- **Beaucoup plus lent que Piper sur CPU** — mesuré dans cet environnement de
  développement (5 phrases courtes en français, CPU seulement, pas de GPU
  disponible pour comparer) : environ **4,2s/extrait pour XTTS contre
  0,18s/extrait pour Piper** (~23x plus lent), plus un chargement du modèle
  d'environ 20s pour XTTS contre 3-4s pour Piper (payé une fois par
  lancement, pas par extrait). Pour de la génération à grand volume sans
  GPU, Piper reste largement plus adapté ; XTTS se justifie surtout pour sa
  meilleure qualité/le clonage de voix, ou avec un GPU pour compenser sa
  lenteur sur CPU.

**Installation :** contrairement à Tortoise, `coqui-tts` n'a pas de pins de
dépendances anciennes et s'installe proprement dans le `.venv` principal
(Python 3.12), en réutilisant le `torch` déjà installé à l'étape 1 :

```bash
uv pip install coqui-tts
```

Avec torch >= 2.9 (la version installée par l'étape 1 au moment de la
rédaction), `coqui-tts` a aussi besoin de l'extra `codec` pour la lecture/
écriture audio (sinon `import TTS` échoue avec une erreur explicite qui vous
redirige ici) :

```bash
uv pip install "coqui-tts[codec]"
```

**Licence et accord des CGU :** les poids XTTS-v2 sont sous la Coqui Public
Model License (CPML) — usage gratuit pour du test/évaluation/recherche
non-commerciale uniquement ; un usage commercial (y compris une génération
à grande échelle destinée à un produit) nécessite une licence séparée
(licensing@coqui.ai). Vérifiez les termes exacts sur
[coqui.ai/cpml](https://coqui.ai/cpml) avant un usage commercial.

Le premier chargement du modèle demande normalement une confirmation
interactive `[y/n]` des CGU, ce qui planterait sous `nohup`. Ce script
l'exige donc en amont via la variable d'environnement `COQUI_TOS_AGREED=1`
(rejeté avec un message clair si absente) plutôt que de laisser le prompt
interactif planter au milieu d'un batch :

```bash
export COQUI_TOS_AGREED=1
python generate_and_concat.py input.example.txt --model xtts --voice "Claribel Dervla"
```

Ou, comme pour `CARTESIA_API_KEY`, ajoutez `COQUI_TOS_AGREED=1` à votre
fichier `.env` pour ne pas avoir à la redéfinir à chaque session.

**Testé de bout en bout :** ce moteur a été vérifié dans cet environnement de
développement avec un téléchargement complet des poids (~2 Go) et une
génération réelle (`--voice "Claribel Dervla" --language fr`), produisant un
`.wav` 24 kHz valide. `--all-voices` (qui boucle sur le catalogue de voix
intégrées) n'a en revanche été vérifié qu'au niveau du code, pas exécuté sur
l'ensemble des ~58 voix — testez avec `--voice-limit` avant un lancement
complet.

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
  `tortoise.py`, `breeze.py`, `piper.py`, `kokoro.py`, `xtts.py`, `cartesia.py`), chacun possédant ses propres
  options CLI, sa validation d'arguments et sa logique de génération.
  `base.py` documente l'interface qu'un nouveau moteur doit implémenter.
  `piper_voice_licenses.json` est l'audit de licences par voix consommé par
  `--piper-commercial-safe` (voir [PIPER_VOICE_LICENSES.md](PIPER_VOICE_LICENSES.md)
  pour le rapport complet et la méthodologie ; le JSON en est dérivé et
  n'a pas vocation à être modifié à la main).
- `tts_batch/cli.py` — assemble le parseur argparse à partir des options
  communes et de celles de chaque moteur, et délègue la validation au
  moteur sélectionné.
- `tts_batch/interactive.py` — l'assistant de configuration guidée (voir
  « Lancer le script » ci-dessus).
- `tts_batch/runner.py` — exécute une commande entièrement analysée/validée.

Ajouter un moteur supplémentaire consiste à créer un nouveau fichier dans
`tts_batch/backends/` implémentant la forme documentée dans `base.py`, puis
à l'ajouter au dictionnaire `BACKENDS` dans
`tts_batch/backends/__init__.py` — rien d'autre n'a besoin de changer.
