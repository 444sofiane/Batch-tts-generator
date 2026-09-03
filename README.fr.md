# Générateur de TTS par lot

Génère plusieurs extraits audio via le modèle TTS PyTorch de Kyutai
(`kyutai-labs/delayed-streams-modeling`) et les concatène par groupe en
fichiers `.wav`.

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
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **GPU NVIDIA** : vérifiez la version CUDA maximale supportée par votre
  pilote avec `nvidia-smi`, puis choisissez une version de torch égale ou
  *inférieure* — une version plus récente que ce que supporte le pilote
  échouera à l'initialisation. Par exemple, pour un pilote supportant
  jusqu'à CUDA 12.7 :
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```

## 2. Rédiger votre fichier d'entrée

Voir `input.example.txt`. Une ligne `# nom` démarre un groupe ; les lignes
non vides qui suivent sont synthétisées dans l'ordre puis concaténées dans
`output/<nom>.wav`. Une ligne vide termine le groupe.

## 3. Lancer le script

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
