# TTS batch generator

Generates multiple TTS clips and concatenates them per group into `.wav`
files. Supports four backends: Kyutai's PyTorch TTS model
(`kyutai-labs/delayed-streams-modeling`, the default), [Tortoise-TTS](https://huggingface.co/spaces/Manmay/tortoise-tts)
(via `--model tortoise`), [Breeze-TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
(via `--model breeze`), and [Cartesia](https://play.cartesia.ai/text-to-speech)'s
cloud API (via `--model cartesia`).

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

## Using the Tortoise-TTS backend

Pass `--model tortoise` to switch from Kyutai to
[Tortoise-TTS](https://huggingface.co/spaces/Manmay/tortoise-tts). It's a
separate model with different tradeoffs:

- **English only** — combining `--model tortoise` with `--language fr` is
  rejected.
- **Voices are built-in presets**, not an HF voice repo path: pass a preset
  name to `--voice` (e.g. `tom`, `angie`, `lj`; see `tortoise/voices/` in
  the installed package for the full list).
- **Much slower than Kyutai**, especially on CPU — a GPU is strongly
  recommended. `--tortoise-preset` controls the quality/speed tradeoff:
  `ultra_fast`, `fast` (default), `standard`, or `high_quality`.
- `--all-voices` works the same way as for Kyutai, but loops over
  Tortoise's built-in preset voices instead of the HF voice repo.
  `--all-fr`/`--all-eng` don't apply (Tortoise voices aren't split by
  language) and are rejected with `--model tortoise`.

### How the Tortoise model works

`TextToSpeech()` (in `load_tortoise_tts()`) doesn't take a model choice —
it always loads one fixed set of pretrained weights from the Hugging Face
repo [`Manmay/tortoise-tts`](https://huggingface.co/spaces/Manmay/tortoise-tts),
cached under `~/.cache/tortoise/models` after the first run. It's a
pipeline of several networks rather than a single model:

- **`autoregressive.pth`** — the core model; turns the input text into a
  sequence of speech tokens, conditioned on the voice samples you provide.
- **`clvp2.pth`** (and optionally `cvvp.pth`) — score multiple candidate
  outputs and keep the ones that best match the text and the target voice.
- **`diffusion_decoder.pth`** — turns the chosen speech tokens into a mel
  spectrogram through a diffusion process. This is the slow step;
  `--tortoise-preset` controls how many diffusion steps it runs
  (`ultra_fast` = fewest/lowest quality, `high_quality` = most/slowest).
- **`vocoder.pth`** — converts the mel spectrogram into the final
  waveform.

So `--voice` and `--tortoise-preset` are the only two things this script
lets you change — which conditioning samples go in, and how much compute
the diffusion decoder spends. The weights themselves aren't swappable
without editing `load_tortoise_tts()` to pass a custom `models_dir`.

**Install:** Tortoise-TTS is a separate, heavier dependency not included in
`requirements.txt`, and pins an old `transformers`/`tokenizers` that has no
prebuilt wheel for Python 3.12 (the main `.venv`) — installing it there either
fails outright or requires building `tokenizers` from source (a Rust
toolchain, and even then old/new dependency versions can conflict). The
straightforward fix is a **separate Python 3.11 venv** just for Tortoise,
since `tokenizers` does have a prebuilt 3.11 wheel:

```bash
uv venv --python 3.11 .venv-tortoise
uv pip install -r requirements.txt --python .venv-tortoise/Scripts/python.exe
uv pip install torch --index-url https://download.pytorch.org/whl/cpu --python .venv-tortoise/Scripts/python.exe
uv pip install tortoise-tts --python .venv-tortoise/Scripts/python.exe
# torchaudio isn't declared as a tortoise-tts dependency but is required at
# import time — install it pinned to the same version as the torch above,
# otherwise you'll hit a native-extension load error:
uv pip install "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cpu --python .venv-tortoise/Scripts/python.exe
```

(Swap the `torch`/`torchaudio` index URL and pin for a CUDA build if you have
a GPU — see the CPU/GPU install step above — matching the exact torch version
you install.)

```bash
.venv-tortoise/Scripts/python.exe generate_and_concat.py input.example.txt --model tortoise --voice tom --device cuda
```

## Using the Breeze-TTS backend

Pass `--model breeze` to use
[Breeze-TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2), a strong
open-weight model — but a heavier, Linux/GPU-only integration:

- **Linux + CUDA GPU only** (developed/tested on an NVIDIA 4090-class GPU;
  no CPU path). It won't run on this project's Windows dev setup — it's
  meant for a Linux server.
- **English/Chinese only** — `--language fr` is rejected with `--model breeze`.
- **Not pip-installable**: you clone the
  [breeze-tts](https://github.com/breezeblue-ai/breeze-tts) repo and its
  `breeze_infer`/`models` packages are imported directly from that checkout
  — point `--breeze-repo-dir` at it, and `--breeze-model-dir` at the
  downloaded model weights (a separate download from Hugging Face).
- **No voice catalog** — `--voice`/`--all-voices`/`--all-fr`/`--all-eng` are
  all rejected with `--model breeze`. Instead, a voice is either:
  - **cloned** from a reference clip: `--breeze-ref-audio ref.wav
    --breeze-ref-text "exact transcript of ref.wav"`
  - **designed** from a text description, no reference audio: just
    `--breeze-instruction "a calm, low-pitched voice"`
  - or both together (**voice direction**): ref audio/text *plus* an
    instruction, which clones the reference speaker's identity but
    overlays the instruction's delivery.
- `--breeze-cfg-scale` (default `1.0`, matching `infer.py`'s own default —
  Breeze's own usage examples use `4` for voice design/direction) and
  `--breeze-seed` (default `42`) tune generation; `--breeze-fast` enables
  Breeze's warmup/CUDA-graph fast path.

**Install** (on the Linux GPU server, not this dev machine):

```bash
git clone https://github.com/breezeblue-ai/breeze-tts.git
cd breeze-tts && python -m pip install -r requirements.txt && cd ..
hf download BreezeBlue/Breeze-TTS-2 --local-dir breeze-tts-2-weights
```

```bash
python generate_and_concat.py input.example.txt --model breeze \
  --breeze-repo-dir breeze-tts --breeze-model-dir breeze-tts-2-weights \
  --breeze-ref-audio ref.wav --breeze-ref-text "Exact transcript of ref.wav"
```

**Note:** Breeze-TTS-2's model weights are under BreezeBlue's Research and
Non-Commercial License (the inference code itself is Apache 2.0) — commercial
use needs a paid subscription through breezeblue.ai.

**Testing caveat:** this backend was implemented from Breeze's published
source (`infer.py`) but not run end-to-end, since it requires Linux + a CUDA
GPU this dev environment doesn't have. Smoke-test it on your server before
relying on it.

## Using the Cartesia backend

Pass `--model cartesia` to use [Cartesia](https://play.cartesia.ai/text-to-speech)'s
Sonic models. Unlike the other three backends, it's a **cloud API, not a
local model** — no download, no GPU, but it does need an account and calls
their servers for every clip:

- **Requires a `CARTESIA_API_KEY`** environment variable. Get one at
  [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) and set it
  before running (lasts for the current terminal session only):
  ```powershell
  # PowerShell
  $env:CARTESIA_API_KEY = "sk_car_..."
  ```
  ```bash
  # bash
  export CARTESIA_API_KEY=sk_car_...
  ```
  Missing/unset is checked upfront and rejected with a clear error before
  any clips are generated.
- **Voices are your Cartesia voice library**, not a filename or preset name:
  pass a voice's `voice_id` (a UUID) to `--voice`. Find one by picking a
  voice on [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) and
  copying its id, or via `client.voices.list()` in the SDK.
- **English and French** are both supported — `--language` is passed
  straight through to Cartesia's API instead of only picking a default
  voice like it does for Kyutai. (Cartesia's own API accepts other language
  codes too, but this script's `--language` flag is currently limited to
  `en`/`fr`, same as the other backends.)
- `--cartesia-model` picks the model (default `sonic-2`, a pinned stable
  release rather than a moving `sonic-latest` alias, so a batch generated
  today sounds the same if you regenerate it later). See
  [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) for other
  available models (e.g. `sonic-turbo` for lower latency).
- `--all-voices` loops over every voice in your Cartesia account instead of
  a fixed catalog. **Each clip is a paid API call** — the pre-run prompt
  warns about account credits/quota instead of printing a time estimate,
  but there's no dollar estimate built in; check
  [play.cartesia.ai](https://play.cartesia.ai/text-to-speech) for current
  pricing before running this at scale. `--all-fr`/`--all-eng` don't apply
  and are rejected with `--model cartesia`.

**Install:** the `cartesia` package is a separate dependency not included in
`requirements.txt` (a lightweight API client, no heavy ML dependencies —
installs cleanly into the main `.venv`, no separate venv needed):

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

**Testing caveat:** this backend was implemented and validated against
Cartesia's published Python SDK source/examples (import shape, `tts.generate_sse`
parameters, `voices.list()`), but not run against the live API, since that
requires a Cartesia account/API key this dev environment doesn't have.
Smoke-test a single clip before running `--all-voices`.
