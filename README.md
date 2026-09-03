# TTS batch generator

Generates multiple TTS clips using Kyutai's PyTorch TTS model
(`kyutai-labs/delayed-streams-modeling`) and concatenates them per group
into `.wav` files.

## 1. Install dependencies

Requires Python 3.12. If you don't have it, [uv](https://docs.astral.sh/uv/)
can fetch it for you without touching your system Python:

```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

(Or with plain pip on an existing Python 3.12: `pip install -r requirements.txt`)

**Then install `torch` separately**, matching your hardware (it's not in
`requirements.txt` because the right build depends on whether you have a
GPU, and if so which CUDA version your driver supports):

- **CPU only** (no GPU, or a GPU too old for CUDA):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **NVIDIA GPU**: check your driver's max supported CUDA version with
  `nvidia-smi`, then pick a torch build at or *below* that version — a
  build newer than your driver supports will fail to initialize. E.g. for
  a driver supporting up to CUDA 12.7:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```

## 2. Write your input file

See `input.example.txt`. A `# name` line starts a group; the non-blank lines
under it are synthesized in order and concatenated into `output/<name>.wav`.
A blank line ends the group.

## 3. Run it

```bash
python generate_and_concat.py input.example.txt --output-dir output --gap-ms 300
```

The first run downloads the model weights from Hugging Face (a few GB,
cached afterward). On CPU, generation is slow: expect roughly 30-45 seconds
per short sentence, regardless of caching — that doesn't improve on later
runs. A compatible GPU (see above) is much faster via `--device cuda`.

## Options

- `--language` — `en` (default) or `fr`. The model itself
  (`kyutai/tts-1.6b-en_fr`) is a single bilingual model that handles
  whatever language the input text is in — this flag only picks a
  matching default voice (an English or French speaker), it doesn't
  switch models.
- `--voice` — voice to use (see `kyutai/tts-voices` on Hugging Face for the
  available options), relative path under that repo. Overrides `--language`.
- `--device` — `cpu` (default) or `cuda` if you installed a matching
  CUDA build of torch.
- `--gap-ms` — silence inserted between clips within a group (default 300ms).

## Generating with every voice

For building a training dataset across many speakers, three flags run your
input against a whole set of voices instead of just one:

- `--all-voices` — every voice in `kyutai/tts-voices` (901+ files).
- `--all-fr` — just the French voices (`cml-tts/fr/`, ~70 files).
- `--all-eng` — just the English Expresso voices (`expresso/`, ~103 files).

These ignore `--voice`/`--language`. Output is flat, not nested per voice:
`<output-dir>/<group><e if the voice is an "_enhanced" variant><voice
index>.wav` — e.g. `intro1.wav`, `introe2.wav`. A `voices_manifest.txt` is
written alongside, mapping each index back to its source voice path.

Since this can mean hundreds of voices x every clip in your input, it prints
a rough time estimate first and asks for confirmation — pass `--yes` to skip
that (needed if you're running non-interactively, e.g. under `nohup`). Use
`--voice-limit N` to try it on just the first N voices before committing to
a full run. It's resumable: rerunning the same command only fills in output
files that don't exist yet, so an interrupted run doesn't lose progress.

```bash
nohup python generate_and_concat.py input.txt --output-dir output --all-fr --device cuda --yes > run.log 2>&1 &
```
