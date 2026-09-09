# Package Restructure + Interactive Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `generate_and_concat.py` (810 lines, four TTS backends in one file) into a `tts_batch` package with one file per concern/backend, and add a no-arguments interactive wizard, without changing any existing `--flag` behavior.

**Architecture:** `generate_and_concat.py` becomes a 4-line entry point that calls `tts_batch.cli.main()`. `cli.py` assembles an argparse parser from shared flags plus each backend module's own (`add_cli_arguments`), and dispatches validation to the selected backend (`validate_args`). `runner.py` dispatches a validated `args.Namespace` to the selected backend's `run_single_voice`/`run_all_voices`. `interactive.py` runs only when the script is invoked with zero CLI arguments: it prompts for the same information the flags would provide, builds an equivalent argv list, and feeds it through the *same* parser/validation — so there is exactly one place argument rules live.

**Tech Stack:** Python 3.12, argparse, stdlib `input()`/`print()` for the wizard (no new dependencies). Existing dependencies unchanged: `moshi`, `sphn`, `numpy`, `tqdm`, `torch`, plus each backend's own optional/lazy dependency (`tortoise-tts`, `breeze_infer` via `--breeze-repo-dir`, `cartesia`).

**Spec:** `docs/superpowers/specs/2026-09-09-package-restructure-and-wizard-design.md`

## Global Constraints

- No new runtime dependencies — the wizard uses only `input()`/`print()`.
- No behavior change to any existing `--flag` combination or its `parser.error(...)` message text — every command in README.md/README.fr.md must keep working identically.
- `tortoise.py` and `breeze.py` must not import their optional third-party packages (`tortoise.api`, `breeze_infer`, `models.*`) at module level — only lazily inside the functions that need them, exactly like the current code. `tts_batch.backends` (via its `__init__.py`) imports all four backend modules eagerly, so a module-level heavy import would break `import tts_batch.cli` in an environment that only has the kyutai dependencies installed (the main `.venv` in this repo).
- No automated test suite is being added — verify with real commands (import checks, a real end-to-end run, a simulated wizard run), per the spec's testing plan.
- `--help` flag *grouping* changes deliberately (backend-specific flags move together, after all shared flags, instead of interleaved as today) — this is a disclosed cosmetic change, not a behavior change. Every flag, its choices, its default, and its `parser.error()` text must still match.

---

## Task 1: Package skeleton + shared modules (input parsing, audio, backend protocol)

**Files:**
- Create: `tts_batch/__init__.py`
- Create: `tts_batch/input_parsing.py`
- Create: `tts_batch/audio.py`
- Create: `tts_batch/backends/__init__.py` (placeholder — replaced in Task 5)
- Create: `tts_batch/backends/base.py`

**Interfaces:**
- Produces: `input_parsing.parse_input(path: Path) -> list[tuple[str, list[str]]]`, `input_parsing.GROUP_HEADER_RE`
- Produces: `audio.concatenate_clips(clips: list[np.ndarray], sample_rate: int, gap_ms: float) -> np.ndarray`
- Produces: `audio.generate_groups_for_voice(synthesize_fn, sample_rate, groups, output_dir, gap_ms, filename_suffix="") -> None`
- Produces: `audio.write_voices_manifest(output_dir: Path, voice_files: list[str]) -> None`
- Produces: `audio.confirm_or_abort(message: str, yes: bool) -> bool`
- Produces: `backends.base.Backend` (a `typing.Protocol`, documentation only)

- [ ] **Step 1: Create the package marker**

`tts_batch/__init__.py`:

```python
"""Batch TTS generator: parses grouped input text, synthesizes each line with
a pluggable backend, and concatenates each group into one .wav file."""
```

- [ ] **Step 2: Create `input_parsing.py`**

`tts_batch/input_parsing.py`:

```python
import re
from pathlib import Path

GROUP_HEADER_RE = re.compile(r"^#\s*(.+)$")


def parse_input(path: Path) -> list[tuple[str, list[str]]]:
    """Split the input file into (group_name, [clip_text, ...]) blocks.

    A line starting with '#' names the group that follows. Consecutive
    non-blank lines after it are the clips, synthesized and concatenated
    in order. A blank line ends the current group.
    """
    groups: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_name is not None and current_lines:
            groups.append((current_name, current_lines.copy()))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            current_lines.clear()
            current_name = None
            continue

        header = GROUP_HEADER_RE.match(line)
        if header:
            flush()
            current_lines.clear()
            current_name = header.group(1).strip()
        elif current_name is not None:
            current_lines.append(line)
        else:
            raise ValueError(
                f"Line {raw_line!r} has no group header (expected a '# name' line first)"
            )

    flush()
    return groups
```

- [ ] **Step 3: Create `audio.py`**

`tts_batch/audio.py`:

```python
from pathlib import Path
from typing import Callable

import numpy as np
import sphn
import tqdm


def concatenate_clips(clips: list[np.ndarray], sample_rate: int, gap_ms: float) -> np.ndarray:
    """Join a group's synthesized clips with a fixed silence gap between them."""
    if not clips:
        return np.zeros(0, dtype=np.float32)

    gap_samples = int(sample_rate * gap_ms / 1000)
    silence = np.zeros(gap_samples, dtype=np.float32)

    pieces = [clips[0]]
    for clip in clips[1:]:
        pieces.append(silence)
        pieces.append(clip)

    return np.concatenate(pieces)


def generate_groups_for_voice(
    synthesize_fn: Callable[[str], np.ndarray],
    sample_rate: int,
    groups: list[tuple[str, list[str]]],
    output_dir: Path,
    gap_ms: float,
    filename_suffix: str = "",
) -> None:
    """Generate+concatenate every group for one voice, skipping already-done groups.

    Backend-agnostic: `synthesize_fn` does the actual per-line generation, so
    this same resumable/skip logic is shared by every backend (see each
    backend's run_single_voice/run_all_voices).

    Skipping existing output files (rather than always overwriting) is what
    makes an interrupted --all-voices run resumable: rerunning the same
    command only fills in what's missing instead of starting over.

    `filename_suffix` disambiguates multiple voices writing into the same
    flat `output_dir` (see each backend's run_all_voices); it's empty for
    the single-voice case, where <group>.wav alone is unambiguous.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name, lines in tqdm.tqdm(groups, desc="groups", leave=False):
        out_path = output_dir / f"{group_name}{filename_suffix}.wav"
        if out_path.exists():
            continue

        clips = [
            synthesize_fn(text)
            for text in tqdm.tqdm(lines, desc=group_name, leave=False)
        ]
        audio = concatenate_clips(clips, sample_rate, gap_ms)
        sphn.write_wav(str(out_path), audio, sample_rate)
        print(f"Wrote {out_path} ({len(audio) / sample_rate:.1f}s)")


def write_voices_manifest(output_dir: Path, voice_files: list[str]) -> None:
    """Index -> source voice path, since flat multi-voice filenames (e.g.
    'intro47.wav') can't carry that information themselves."""
    manifest_path = output_dir / "voices_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for index, voice_file in enumerate(voice_files, start=1):
            manifest.write(f"{index}\t{voice_file}\n")
    print(f"Wrote voice index -> source mapping to {manifest_path}")


def confirm_or_abort(message: str, yes: bool) -> bool:
    """Print `message`, then ask to continue unless `yes`.

    Shared by every backend's run_all_voices: a big run (hundreds of voices,
    hours of compute, or paid API calls) prints an estimate/cost note and
    waits for confirmation, unless --yes was passed (needed for
    non-interactive runs, e.g. under nohup). Returns True if the run should
    proceed.
    """
    print(message)
    if yes:
        return True
    if input("Continue? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return False
    return True
```

- [ ] **Step 4: Create the backend placeholder package**

`tts_batch/backends/__init__.py` (replaced with the real registry in Task 5):

```python
"""TTS backend registry — populated once every backend module exists (see
Task 5 of the restructure plan)."""
```

- [ ] **Step 5: Create the backend Protocol**

`tts_batch/backends/base.py`:

```python
"""Documents the interface every tts_batch.backends.<name> module implements.

Backend modules (kyutai.py, tortoise.py, breeze.py, cartesia.py) are used as
modules, not class instances, so this Protocol has no `self` - it describes
the plain functions/attributes each module exposes at its top level. It's
not enforced at runtime (nothing calls isinstance against it); it exists so
a reader - human or type checker - has one place to see the whole shape
expected of a backend, e.g. when adding a new one.
"""

import argparse
from typing import Callable, Protocol


class Backend(Protocol):
    #: Value used for --model and as the key in backends.BACKENDS.
    NAME: str
    #: One-line description shown in --help and the interactive wizard's
    #: backend menu.
    DESCRIPTION: str

    #: Registers this backend's own CLI flags (e.g. --tortoise-preset) on
    #: the shared parser. Flags used by more than one backend (--voice,
    #: --language, --device, etc.) are defined once in cli.py instead.
    add_cli_arguments: Callable[[argparse.ArgumentParser], None]

    #: Backend-specific argument validation, called after argparse itself
    #: has parsed `args`. Reports problems via parser.error(...), same as
    #: argparse's own validation.
    validate_args: Callable[[argparse.Namespace, argparse.ArgumentParser], None]

    #: Generates+concatenates every group in `groups` for the one voice
    #: selected by `args`.
    run_single_voice: Callable[[argparse.Namespace, list[tuple[str, list[str]]]], None]

    #: Generates+concatenates every group in `groups` for every voice this
    #: backend knows about (--all-voices and friends). Backends with no
    #: voice catalog (breeze) implement this to raise NotImplementedError,
    #: since validate_args rejects --all-voices for them before this would
    #: ever run.
    run_all_voices: Callable[[argparse.Namespace, list[tuple[str, list[str]]]], None]

    #: Prompts the user (via input()) for this backend's own settings in
    #: the interactive wizard, returning them as an argv fragment - e.g.
    #: ["--voice", "tom", "--tortoise-preset", "fast"]. `common` is
    #: reserved for context shared across backends; none is passed yet.
    interactive_args: Callable[[dict], list[str]]
```

- [ ] **Step 6: Verify**

Run:

```bash
python -c "from pathlib import Path; from tts_batch.input_parsing import parse_input; print(parse_input(Path('input.example.txt')))"
python -c "from tts_batch.audio import concatenate_clips; import numpy as np; c = concatenate_clips([np.ones(3, dtype='float32'), np.ones(2, dtype='float32')], 10, 100); print(c.shape, c.dtype)"
python -c "import tts_batch.backends.base"
```

Expected: first command prints `[('intro', ['Welcome to this example.', 'It shows how lines are grouped into one output file.']), ('chapter_one', ['This is the first clip of chapter one.', 'And this is the second clip, appended right after it.'])]`; second prints `(6,) float32` (3 + 1 gap sample + 2, since `10 * 100 / 1000 = 1`); third prints nothing (no error).

- [ ] **Step 7: Commit**

```bash
git add tts_batch/__init__.py tts_batch/input_parsing.py tts_batch/audio.py tts_batch/backends/__init__.py tts_batch/backends/base.py
git commit -m "ADD: tts_batch package skeleton with shared input/audio modules"
```

---

## Task 2: Kyutai backend module

**Files:**
- Create: `tts_batch/backends/kyutai.py`

**Interfaces:**
- Consumes: `tts_batch.audio.generate_groups_for_voice`, `.write_voices_manifest`, `.confirm_or_abort` (Task 1)
- Produces: `kyutai.NAME`, `kyutai.DESCRIPTION`, `kyutai.add_cli_arguments(parser)`, `kyutai.validate_args(args, parser)`, `kyutai.run_single_voice(args, groups)`, `kyutai.run_all_voices(args, groups)`, `kyutai.interactive_args(common) -> list[str]` — matching the `backends.base.Backend` protocol from Task 1
- Produces (module-internal, used by Task 7's cli.py `--all-fr`/`--all-eng` help text): `kyutai.FR_VOICES_PREFIX`, `kyutai.EN_VOICES_PREFIX`

- [ ] **Step 1: Create `kyutai.py`**

`tts_batch/backends/kyutai.py`:

```python
"""Kyutai's PyTorch TTS backend (the default) - kyutai-labs/delayed-streams-modeling.

A single bilingual (English/French) model that runs locally on CPU or GPU.
The language of the output is just whatever language the input text is in;
--language only picks a default voice recording to match.
"""

import functools

import numpy as np
import torch
import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

from .. import audio

NAME = "kyutai"
DESCRIPTION = "Kyutai's PyTorch TTS model (bilingual EN/FR, runs locally on CPU or GPU) - the default"

# kyutai/tts-1.6b-en_fr is a single bilingual model; the language of the
# output is just whatever language the input text is in. What actually
# differs per language is the voice recording, so --language only picks a
# default voice - pass --voice directly to use any other one.
DEFAULT_VOICE_BY_LANGUAGE = {
    "en": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "fr": "cml-tts/fr/10087_11650_000028-0002.wav",
}

FR_VOICES_PREFIX = "cml-tts/fr/"
EN_VOICES_PREFIX = "expresso/"

# Rough per-clip generation time used only to warn before a long
# --all-voices run; actual time varies a lot with sentence length and the
# specific hardware, this is just a ballpark to size the estimate.
ROUGH_SECONDS_PER_CLIP_CPU = 40
ROUGH_SECONDS_PER_CLIP_GPU = 2


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_DSM_TTS_REPO,
        help="HF repo for the TTS model (--model kyutai only)",
    )


def validate_args(args, parser) -> None:
    pass


def load_model(args) -> TTSModel:
    print("Loading model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
    return TTSModel.from_checkpoint_info(checkpoint_info, n_q=32, temp=0.6, device=args.device)


def synthesize_clip(tts_model: TTSModel, condition_attributes, text: str) -> np.ndarray:
    """Run one line of text through the loaded model and return float32 PCM."""
    entries = tts_model.prepare_script([text], padding_between=1)
    result = tts_model.generate([entries], [condition_attributes])

    with tts_model.mimi.streaming(1), torch.no_grad():
        pcms = []
        for frame in result.frames[tts_model.delay_steps :]:
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))
        return np.concatenate(pcms, axis=-1).astype(np.float32)


def list_all_voices(voice_repo: str, prefix: str | None = None) -> list[str]:
    """Every voice embedding file in the voice repo, as full repo-relative paths.

    If `prefix` is given, only files under that path prefix are returned
    (e.g. "cml-tts/fr/" for just the French voices).
    """
    files = list_repo_files(voice_repo)
    voices = (f for f in files if f.endswith(".safetensors"))
    if prefix is not None:
        voices = (f for f in voices if f.startswith(prefix))
    return sorted(voices)


def run_single_voice(args, groups) -> None:
    tts_model = load_model(args)
    voice = args.voice or DEFAULT_VOICE_BY_LANGUAGE[args.language]
    voice_path = voice if voice.endswith(".safetensors") else tts_model.get_voice_path(voice)
    condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
    synthesize_fn = functools.partial(synthesize_clip, tts_model, condition_attributes)
    audio.generate_groups_for_voice(
        synthesize_fn, tts_model.mimi.sample_rate, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    tts_model = load_model(args)

    if args.all_fr:
        voice_prefix = FR_VOICES_PREFIX
    elif args.all_eng:
        voice_prefix = EN_VOICES_PREFIX
    else:
        voice_prefix = None
    voice_files = list_all_voices(tts_model.voice_repo, prefix=voice_prefix)
    if args.voice_limit:
        voice_files = voice_files[: args.voice_limit]

    seconds_per_clip = (
        ROUGH_SECONDS_PER_CLIP_CPU if args.device == "cpu" else ROUGH_SECONDS_PER_CLIP_GPU
    )
    clips_per_voice = sum(len(lines) for _, lines in groups)
    total_clips = clips_per_voice * len(voice_files)
    estimate_hours = total_clips * seconds_per_clip / 3600
    message = (
        f"{len(voice_files)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Rough estimate at ~{seconds_per_clip}s/clip: {estimate_hours:.1f} hours "
        f"(device={args.device}, very approximate)."
    )
    if not audio.confirm_or_abort(message, args.yes):
        return

    # All voices write into the same flat output_dir (group.wav alone would
    # collide across voices), so filenames get a sequential per-voice index
    # instead, and a manifest records what each index actually maps to.
    audio.write_voices_manifest(args.output_dir, voice_files)

    for index, voice_file in enumerate(tqdm.tqdm(voice_files, desc="voices"), start=1):
        try:
            voice_path = hf_hub_download(tts_model.voice_repo, voice_file)
            condition_attributes = tts_model.make_condition_attributes(
                [voice_path], cfg_coef=2.0
            )
            # voice_suffix (e.g. ".1e68beda@240.safetensors") is the same
            # constant for every file for this model, so checking for
            # "_enhanced" this way works regardless of voice.
            voice_base = voice_file.removesuffix(tts_model.voice_suffix)
            marker = "e" if voice_base.endswith("_enhanced.wav") else ""
            synthesize_fn = functools.partial(synthesize_clip, tts_model, condition_attributes)
            audio.generate_groups_for_voice(
                synthesize_fn,
                tts_model.mimi.sample_rate,
                groups,
                args.output_dir,
                args.gap_ms,
                filename_suffix=f"{marker}{index}",
            )
        except Exception as e:
            # A single bad/unusual voice file (e.g. among the less-curated
            # voice-donations) shouldn't abort a run spanning hundreds of
            # others and many hours.
            print(f"Skipping voice {voice_file!r} after error: {e!r}")


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Kyutai options --")
    print("  [1] One voice (default)")
    print("  [2] All voices in the voice repo")
    print("  [3] All French voices only")
    print("  [4] All English voices only")
    mode = input("Choose [1-4] (default 1): ").strip() or "1"

    if mode == "2":
        argv.append("--all-voices")
    elif mode == "3":
        argv.append("--all-fr")
    elif mode == "4":
        argv.append("--all-eng")
    else:
        language = input("Language, en or fr [en]: ").strip() or "en"
        argv += ["--language", language]
        voice = input(
            "Voice path (leave blank to use the default voice for the language): "
        ).strip()
        if voice:
            argv += ["--voice", voice]

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
```

- [ ] **Step 2: Verify**

Run:

```bash
python -c "from tts_batch.backends import kyutai; print(kyutai.NAME); print(kyutai.DESCRIPTION); print(kyutai.FR_VOICES_PREFIX, kyutai.EN_VOICES_PREFIX)"
```

Expected: prints `kyutai`, the description string, then `cml-tts/fr/ expresso/`, with no import errors (moshi/torch are already installed in `.venv`).

- [ ] **Step 3: Commit**

```bash
git add tts_batch/backends/kyutai.py
git commit -m "ADD: kyutai backend module"
```

---

## Task 3: Tortoise backend module

**Files:**
- Create: `tts_batch/backends/tortoise.py`

**Interfaces:**
- Consumes: `tts_batch.audio.generate_groups_for_voice`, `.write_voices_manifest`, `.confirm_or_abort` (Task 1)
- Produces: `tortoise.NAME`, `tortoise.DESCRIPTION`, `tortoise.add_cli_arguments`, `tortoise.validate_args`, `tortoise.run_single_voice`, `tortoise.run_all_voices`, `tortoise.interactive_args` — matching `backends.base.Backend`

- [ ] **Step 1: Create `tortoise.py`**

`tts_batch/backends/tortoise.py`:

```python
"""Tortoise-TTS backend (https://huggingface.co/spaces/Manmay/tortoise-tts).

English-only, and requires the separate tortoise-tts package (see README) -
imported lazily so installing it is only needed when --model tortoise is
actually used.
"""

import functools

import numpy as np
import torch
import tqdm

from .. import audio

NAME = "tortoise"
DESCRIPTION = (
    "Tortoise-TTS (English only, built-in preset voices, requires the separate "
    "tortoise-tts package - see README)"
)

# Tortoise's output is always 24kHz, regardless of voice or preset.
SAMPLE_RATE = 24000

# Rough per-clip generation time on GPU for each Tortoise quality preset,
# used only to size the --all-voices time estimate (CPU is impractically
# slow for Tortoise and isn't estimated here).
ROUGH_SECONDS_PER_CLIP = {
    "ultra_fast": 15,
    "fast": 30,
    "standard": 90,
    "high_quality": 240,
}


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--tortoise-preset",
        type=str,
        choices=["ultra_fast", "fast", "standard", "high_quality"],
        default="fast",
        help=(
            "Quality/speed tradeoff for --model tortoise (default: fast). Higher "
            "quality is much slower, especially on CPU."
        ),
    )


def validate_args(args, parser) -> None:
    if args.all_fr or args.all_eng:
        parser.error(
            "--all-fr/--all-eng are --model kyutai only (Tortoise voices aren't "
            "split by language); use --all-voices instead."
        )
    if args.language == "fr":
        parser.error("Tortoise-TTS is English-only; --language fr requires --model kyutai.")
    if not args.all_voices and not args.voice:
        parser.error(
            "--model tortoise requires --voice <preset-name> (e.g. 'tom', 'angie'), "
            "or --all-voices to generate every built-in preset."
        )


def load_tortoise_tts(device: str):
    """Lazily construct the Tortoise TextToSpeech model.

    Imported here rather than at module scope so that installing
    tortoise-tts (a separate, heavier dependency - see README) is only
    required when --model tortoise is actually used.
    """
    from tortoise.api import TextToSpeech

    return TextToSpeech(device=device)


def synthesize_clip(tts, voice_samples, conditioning_latents, preset: str, text: str) -> np.ndarray:
    """Run one line of text through a loaded Tortoise model and return float32 PCM."""
    with torch.no_grad():
        gen = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset,
        )
    return gen.cpu().numpy()[0, 0].astype(np.float32)


def list_all_voices() -> list[str]:
    """Every built-in Tortoise preset voice name."""
    from tortoise.utils.audio import get_voices

    # "random" is a pseudo-voice (a random blend, non-reproducible) handled
    # by Tortoise's own CLI rather than a real bundled voice directory; it
    # isn't expected to show up here, but is excluded defensively.
    return sorted(name for name in get_voices() if name != "random")


def run_single_voice(args, groups) -> None:
    from tortoise.utils.audio import load_voice

    print("Loading model...")
    tts = load_tortoise_tts(args.device)
    voice_samples, conditioning_latents = load_voice(args.voice)
    synthesize_fn = functools.partial(
        synthesize_clip, tts, voice_samples, conditioning_latents, args.tortoise_preset
    )
    audio.generate_groups_for_voice(
        synthesize_fn, SAMPLE_RATE, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    from tortoise.utils.audio import load_voice

    voice_names = list_all_voices()
    if args.voice_limit:
        voice_names = voice_names[: args.voice_limit]

    seconds_per_clip = ROUGH_SECONDS_PER_CLIP[args.tortoise_preset]
    clips_per_voice = sum(len(lines) for _, lines in groups)
    total_clips = clips_per_voice * len(voice_names)
    estimate_hours = total_clips * seconds_per_clip / 3600
    message = (
        f"{len(voice_names)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Rough estimate at ~{seconds_per_clip}s/clip ('{args.tortoise_preset}' preset): "
        f"{estimate_hours:.1f} hours (device={args.device}, very approximate)."
    )
    if not audio.confirm_or_abort(message, args.yes):
        return

    audio.write_voices_manifest(args.output_dir, voice_names)

    print("Loading model...")
    tts = load_tortoise_tts(args.device)
    for index, voice_name in enumerate(tqdm.tqdm(voice_names, desc="voices"), start=1):
        try:
            voice_samples, conditioning_latents = load_voice(voice_name)
            synthesize_fn = functools.partial(
                synthesize_clip,
                tts,
                voice_samples,
                conditioning_latents,
                args.tortoise_preset,
            )
            audio.generate_groups_for_voice(
                synthesize_fn,
                SAMPLE_RATE,
                groups,
                args.output_dir,
                args.gap_ms,
                filename_suffix=str(index),
            )
        except Exception as e:
            print(f"Skipping voice {voice_name!r} after error: {e!r}")


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Tortoise options --")
    all_voices = input("Generate for every built-in preset voice? [y/N]: ").strip().lower() == "y"
    if all_voices:
        argv.append("--all-voices")
    else:
        voice = input("Voice preset name (e.g. 'tom', 'angie'): ").strip()
        while not voice:
            voice = input("Voice preset name (e.g. 'tom', 'angie'): ").strip()
        argv += ["--voice", voice]

    preset = input(
        "Quality preset, ultra_fast/fast/standard/high_quality [fast]: "
    ).strip() or "fast"
    argv += ["--tortoise-preset", preset]

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
```

- [ ] **Step 2: Verify**

Run:

```bash
python -c "from tts_batch.backends import tortoise; print(tortoise.NAME); print(tortoise.SAMPLE_RATE)"
```

Expected: prints `tortoise` then `24000`, with **no error even though the `tortoise-tts` package is not installed** in the main `.venv` — this confirms the lazy-import pattern is preserved (only `load_tortoise_tts`/`run_single_voice`/`run_all_voices`/`list_all_voices` import `tortoise.*`, not the module top level).

- [ ] **Step 3: Commit**

```bash
git add tts_batch/backends/tortoise.py
git commit -m "ADD: tortoise backend module"
```

---

## Task 4: Breeze backend module

**Files:**
- Create: `tts_batch/backends/breeze.py`

**Interfaces:**
- Consumes: `tts_batch.audio.generate_groups_for_voice` (Task 1)
- Produces: `breeze.NAME`, `breeze.DESCRIPTION`, `breeze.add_cli_arguments`, `breeze.validate_args`, `breeze.run_single_voice`, `breeze.run_all_voices`, `breeze.interactive_args` — matching `backends.base.Backend`

- [ ] **Step 1: Create `breeze.py`**

`tts_batch/backends/breeze.py`:

```python
"""Breeze-TTS 2 backend (https://huggingface.co/BreezeBlue/Breeze-TTS-2).

Linux + CUDA GPU only, English/Chinese, no voice catalog (clone a reference
clip and/or describe a voice instead). Not pip-installable: point
--breeze-repo-dir at a local clone of github.com/breezeblue-ai/breeze-tts,
whose breeze_infer/models packages are imported directly from that checkout
rather than installed - only once --model breeze is actually used, so this
integration doesn't affect the other backends.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import audio

NAME = "breeze"
DESCRIPTION = (
    "Breeze-TTS 2 (English/Chinese, voice cloning/design, requires a local repo "
    "checkout + weights, Linux + CUDA GPU only - see README)"
)

# Breeze's own infer.py hardcodes these rather than exposing them as CLI
# flags, so they're mirrored here as fixed values to match upstream behavior.
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--breeze-repo-dir",
        type=Path,
        default=None,
        help=(
            "Path to a local clone of github.com/breezeblue-ai/breeze-tts (required "
            "for --model breeze; its breeze_infer/models packages are imported "
            "directly rather than pip-installed)."
        ),
    )
    parser.add_argument(
        "--breeze-model-dir",
        type=Path,
        default=None,
        help=(
            "Path to the downloaded BreezeBlue/Breeze-TTS-2 weights (required for "
            "--model breeze)."
        ),
    )
    parser.add_argument(
        "--breeze-ref-audio",
        type=Path,
        default=None,
        help=(
            "Reference voice clip to clone (--model breeze only). Must be given "
            "together with --breeze-ref-text. Omit both for voice-design mode, "
            "where the voice comes purely from --breeze-instruction."
        ),
    )
    parser.add_argument(
        "--breeze-ref-text",
        type=str,
        default=None,
        help="Exact transcript of --breeze-ref-audio (--model breeze only).",
    )
    parser.add_argument(
        "--breeze-instruction",
        type=str,
        default="Speak clearly and naturally.",
        help=(
            "Voice description (without --breeze-ref-audio) or delivery instruction "
            "(with --breeze-ref-audio) for --model breeze. Default matches Breeze's "
            "own infer.py default."
        ),
    )
    parser.add_argument(
        "--breeze-cfg-scale",
        type=float,
        default=1.0,
        help=(
            "Guidance scale for --model breeze (default: 1.0, matching infer.py's "
            "own default - Breeze's own usage examples use 4 for voice design/"
            "direction)."
        ),
    )
    parser.add_argument(
        "--breeze-seed",
        type=int,
        default=42,
        help="Random seed for --model breeze (default: 42, matching infer.py).",
    )
    parser.add_argument(
        "--breeze-fast",
        action="store_true",
        help=(
            "Enable Breeze's fast-path optimizations (warmup + CUDA graphs) for "
            "--model breeze, equivalent to infer.py's --fast-all."
        ),
    )


def validate_args(args, parser) -> None:
    if args.breeze_repo_dir is None or args.breeze_model_dir is None:
        parser.error("--model breeze requires --breeze-repo-dir and --breeze-model-dir.")
    if bool(args.breeze_ref_audio) != bool(args.breeze_ref_text):
        parser.error("--breeze-ref-audio and --breeze-ref-text must be given together.")
    if args.voice or args.all_voices or args.all_fr or args.all_eng:
        parser.error(
            "--voice/--all-voices/--all-fr/--all-eng aren't supported with "
            "--model breeze (there's no voice catalog to select from) - use "
            "--breeze-ref-audio/--breeze-ref-text and/or --breeze-instruction "
            "instead."
        )
    if args.language == "fr":
        parser.error(
            "Breeze-TTS supports English/Chinese, not French; --language fr "
            "requires --model kyutai."
        )


def run_single_voice(args, groups) -> None:
    # breeze-tts isn't pip-installable: you clone the repo and its top-level
    # breeze_infer/models packages are imported directly, so the checkout
    # needs to be on sys.path first.
    sys.path.insert(0, str(args.breeze_repo_dir))

    from breeze_infer.runtime import (
        load_runtime,
        resolve_device,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import get_template, prepare_inputs
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    from models.warmup_profile import load_warmup_profile

    print("Loading model...")
    tokenizer, model, audio_tokenizer = load_runtime(
        args.breeze_model_dir, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=True if args.breeze_fast else None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)

    if runtime.fast_enabled:
        fast_config_path = args.breeze_repo_dir / "configs" / "fast.json"
        profile = load_warmup_profile(fast_config_path)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms")

    # Built once and reused for every clip: which template (plain
    # voice-design vs. reference-clone/direction) and which ref audio/text,
    # if any, don't change line to line, only the text does.
    request_base = {"instruction": args.breeze_instruction, "speaker": "S0"}
    if args.breeze_ref_audio is not None:
        request_base["ref_audio_path"] = str(args.breeze_ref_audio)
        request_base["ref_text"] = args.breeze_ref_text.strip()
        template = get_template("ref_edit_tata")
    else:
        template = get_template("tts_instruction")

    def synthesize_fn(text: str) -> np.ndarray:
        request = {**request_base, "id": "clip", "text": text}
        set_all_seeds(args.breeze_seed)
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            template,
            guidance_scale=args.breeze_cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        chunks = [
            chunk.audio
            for chunk in runtime.iter_audio_chunks(inputs, request_id="clip", seed=args.breeze_seed)
        ]
        return np.concatenate(chunks).astype(np.float32)

    audio.generate_groups_for_voice(
        synthesize_fn, runtime.sample_rate, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    raise NotImplementedError(
        "--model breeze has no voice catalog to loop over; validate_args() rejects "
        "--all-voices/--all-fr/--all-eng before this would ever be called."
    )


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Breeze options --")
    repo_dir = input("Path to your breeze-tts repo checkout: ").strip()
    while not repo_dir:
        repo_dir = input("Path to your breeze-tts repo checkout: ").strip()
    argv += ["--breeze-repo-dir", repo_dir]

    model_dir = input("Path to the downloaded Breeze-TTS-2 weights: ").strip()
    while not model_dir:
        model_dir = input("Path to the downloaded Breeze-TTS-2 weights: ").strip()
    argv += ["--breeze-model-dir", model_dir]

    ref_audio = input(
        "Reference voice clip to clone (leave blank to design a voice instead): "
    ).strip()
    if ref_audio:
        ref_text = input("Exact transcript of that reference clip: ").strip()
        while not ref_text:
            ref_text = input("Exact transcript of that reference clip: ").strip()
        argv += ["--breeze-ref-audio", ref_audio, "--breeze-ref-text", ref_text]

    instruction = input(
        "Voice description or delivery instruction [Speak clearly and naturally.]: "
    ).strip()
    if instruction:
        argv += ["--breeze-instruction", instruction]

    if input("Enable fast-path warmup/CUDA graphs? [y/N]: ").strip().lower() == "y":
        argv.append("--breeze-fast")

    return argv
```

- [ ] **Step 2: Verify**

Run:

```bash
python -c "from tts_batch.backends import breeze; print(breeze.NAME); print(breeze.MAX_NEW_TOKENS)"
```

Expected: prints `breeze` then `1500`, with no error even though `breeze_infer`/`models` aren't installed anywhere in this repo (they're only imported inside `run_single_voice`).

- [ ] **Step 3: Commit**

```bash
git add tts_batch/backends/breeze.py
git commit -m "ADD: breeze backend module"
```

---

## Task 5: Cartesia backend module + finalize the backend registry

**Files:**
- Create: `tts_batch/backends/cartesia.py`
- Modify: `tts_batch/backends/__init__.py` (replace the Task 1 placeholder)

**Interfaces:**
- Consumes: `tts_batch.audio.generate_groups_for_voice`, `.write_voices_manifest`, `.confirm_or_abort` (Task 1)
- Produces: `cartesia.NAME`, `cartesia.DESCRIPTION`, `cartesia.add_cli_arguments`, `cartesia.validate_args`, `cartesia.run_single_voice`, `cartesia.run_all_voices`, `cartesia.interactive_args` — matching `backends.base.Backend`
- Produces: `backends.BACKENDS: dict[str, module]` — consumed by Task 6 (`runner.py`) and Task 7 (`cli.py`, `interactive.py`)

- [ ] **Step 1: Create `cartesia.py`**

`tts_batch/backends/cartesia.py`:

```python
"""Cartesia cloud TTS backend (https://play.cartesia.ai/text-to-speech).

Unlike the other backends, this is a cloud API, not a local model: no
download, no GPU, but it does need a CARTESIA_API_KEY and calls Cartesia's
servers for every clip. The `cartesia` package is imported lazily so it's
only required when --model cartesia is actually used.
"""

import functools
import os

import numpy as np
import tqdm

from .. import audio

NAME = "cartesia"
DESCRIPTION = (
    "Cartesia's cloud API (English/French, no local model or GPU, requires the "
    "cartesia package and CARTESIA_API_KEY - see README)"
)

# Sample rate requested from Cartesia's API for raw PCM output. Cartesia
# supports several rates; 44100 is used consistently so every clip and voice
# shares one rate regardless of which model/voice generated it.
SAMPLE_RATE = 44100


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--cartesia-model",
        type=str,
        default="sonic-2",
        help=(
            "Cartesia model id for --model cartesia (default: 'sonic-2', a pinned "
            "stable model rather than a moving 'latest' alias, since batches should "
            "stay reproducible). See play.cartesia.ai for other available models."
        ),
    )


def validate_args(args, parser) -> None:
    if args.all_fr or args.all_eng:
        parser.error(
            "--all-fr/--all-eng are --model kyutai only; use --all-voices for "
            "--model cartesia (it loops over your Cartesia account's voice "
            "library instead of a language-split repo)."
        )
    if not args.all_voices and not args.voice:
        parser.error(
            "--model cartesia requires --voice <voice_id> (copy one from "
            "play.cartesia.ai), or --all-voices to loop over your account's "
            "whole voice library."
        )
    if not os.environ.get("CARTESIA_API_KEY"):
        parser.error(
            "CARTESIA_API_KEY is not set. Get a key from play.cartesia.ai and "
            "export it as an environment variable before running with "
            "--model cartesia."
        )


def load_client():
    """Lazily construct the Cartesia API client.

    Imported here rather than at module scope so that installing the
    `cartesia` package (a separate, cloud-API-only dependency - see README)
    is only required when --model cartesia is actually used. Assumes
    CARTESIA_API_KEY has already been validated present (see validate_args).
    """
    from cartesia import Cartesia

    return Cartesia(api_key=os.environ["CARTESIA_API_KEY"])


def synthesize_clip(client, voice_id: str, model_id: str, language: str, text: str) -> np.ndarray:
    """Run one line of text through Cartesia's cloud API and return float32 PCM."""
    chunks = []
    stream = client.tts.generate_sse(
        model_id=model_id,
        transcript=text,
        voice=voice_id,
        language=language,
        output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": SAMPLE_RATE,
        },
    )
    for event in stream:
        if event.type == "chunk" and event.audio:
            chunks.append(event.audio)
        elif event.type == "error":
            raise RuntimeError(f"Cartesia error: {event.title}: {event.message}")
    return np.frombuffer(b"".join(chunks), dtype=np.float32)


def list_all_voices(client) -> list[tuple[str, str]]:
    """Every (voice_id, name) pair in this Cartesia account's voice library."""
    return sorted(
        ((voice.id, voice.name) for voice in client.voices.list()),
        key=lambda pair: pair[1],
    )


def run_single_voice(args, groups) -> None:
    client = load_client()
    synthesize_fn = functools.partial(
        synthesize_clip, client, args.voice, args.cartesia_model, args.language
    )
    audio.generate_groups_for_voice(
        synthesize_fn, SAMPLE_RATE, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    client = load_client()
    voices = list_all_voices(client)
    if args.voice_limit:
        voices = voices[: args.voice_limit]

    clips_per_voice = sum(len(lines) for _, lines in groups)
    total_clips = clips_per_voice * len(voices)
    message = (
        f"{len(voices)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Each clip is a call to Cartesia's cloud API ('{args.cartesia_model}' model), "
        "consuming your account's credits/quota - check play.cartesia.ai for current "
        "pricing before running this at scale."
    )
    if not audio.confirm_or_abort(message, args.yes):
        return

    audio.write_voices_manifest(
        args.output_dir, [f"{voice_id} ({name})" for voice_id, name in voices]
    )

    for index, (voice_id, name) in enumerate(tqdm.tqdm(voices, desc="voices"), start=1):
        try:
            synthesize_fn = functools.partial(
                synthesize_clip, client, voice_id, args.cartesia_model, args.language
            )
            audio.generate_groups_for_voice(
                synthesize_fn,
                SAMPLE_RATE,
                groups,
                args.output_dir,
                args.gap_ms,
                filename_suffix=str(index),
            )
        except Exception as e:
            # A single bad/rate-limited voice shouldn't abort a run spanning
            # the whole account's voice library.
            print(f"Skipping voice {name!r} ({voice_id}) after error: {e!r}")


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Cartesia options --")
    print("(Requires CARTESIA_API_KEY to already be set in this environment.)")
    all_voices = (
        input("Generate for every voice in your account's library? [y/N]: ").strip().lower()
        == "y"
    )
    if all_voices:
        argv.append("--all-voices")
    else:
        voice_id = input("Voice ID (copy one from play.cartesia.ai): ").strip()
        while not voice_id:
            voice_id = input("Voice ID (copy one from play.cartesia.ai): ").strip()
        argv += ["--voice", voice_id]

    model = input("Cartesia model id [sonic-2]: ").strip() or "sonic-2"
    argv += ["--cartesia-model", model]
    return argv
```

- [ ] **Step 2: Replace the backend registry placeholder**

`tts_batch/backends/__init__.py`:

```python
"""Registry of every supported TTS backend, keyed by --model name.

Dict order here is display order: it decides both the order backend-
specific flags appear in --help (see cli.build_parser) and the order
choices are numbered in the interactive wizard's backend menu (see
interactive.run_wizard).
"""

from . import breeze, cartesia, kyutai, tortoise

BACKENDS = {
    kyutai.NAME: kyutai,
    tortoise.NAME: tortoise,
    breeze.NAME: breeze,
    cartesia.NAME: cartesia,
}
```

- [ ] **Step 3: Verify**

Run:

```bash
python -c "from tts_batch.backends import BACKENDS; print(list(BACKENDS))"
```

Expected: prints `['kyutai', 'tortoise', 'breeze', 'cartesia']` — same order as the original `--model` choices — with no import errors, confirming all four modules (including the two with optional heavy dependencies not installed here) import cleanly together.

- [ ] **Step 4: Commit**

```bash
git add tts_batch/backends/cartesia.py tts_batch/backends/__init__.py
git commit -m "ADD: cartesia backend module and finalize backend registry"
```

---

## Task 6: runner.py

**Files:**
- Create: `tts_batch/runner.py`

**Interfaces:**
- Consumes: `tts_batch.input_parsing.parse_input` (Task 1), `tts_batch.backends.BACKENDS` (Task 5)
- Produces: `runner.run(args: argparse.Namespace) -> None` — consumed by Task 7's `cli.main()`

- [ ] **Step 1: Create `runner.py`**

`tts_batch/runner.py`:

```python
"""Dispatches a fully-parsed/validated args.Namespace to the selected backend."""

from . import input_parsing
from .backends import BACKENDS


def run(args) -> None:
    groups = input_parsing.parse_input(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = BACKENDS[args.model]
    if args.all_voices or args.all_fr or args.all_eng:
        backend.run_all_voices(args, groups)
    else:
        backend.run_single_voice(args, groups)
```

- [ ] **Step 2: Verify**

Run:

```bash
python -c "from tts_batch import runner; print(runner.run)"
```

Expected: prints something like `<function run at 0x...>`, no import error (confirms `input_parsing` and `backends` wire up correctly).

- [ ] **Step 3: Commit**

```bash
git add tts_batch/runner.py
git commit -m "ADD: runner.py dispatch"
```

---

## Task 7: cli.py + interactive.py + thin entry point

**Files:**
- Create: `tts_batch/cli.py`
- Create: `tts_batch/interactive.py`
- Modify: `generate_and_concat.py` (replace entire 810-line contents with a 4-line shim)

**Interfaces:**
- Consumes: `tts_batch.backends.BACKENDS` (Task 5), `tts_batch.backends.kyutai.FR_VOICES_PREFIX`/`EN_VOICES_PREFIX` (Task 2), `tts_batch.runner.run` (Task 6)
- Produces: `cli.build_parser() -> argparse.ArgumentParser`, `cli.validate_args(args, parser) -> None`, `cli.main() -> None`
- Produces: `interactive.run_wizard(parser: argparse.ArgumentParser) -> list[str]`

(cli.py, interactive.py, and the entry point are grouped into one task because they form a closed loop — `cli.main()` calls `interactive.run_wizard()` and `runner.run()`, and the entry point calls `cli.main()` — so none of the three can be meaningfully verified alone; `runner.py` from Task 6 already exists, so `cli.py`'s module-level `from . import interactive, runner` resolves cleanly as soon as `interactive.py` is created in Step 1 below.)

- [ ] **Step 1: Create `interactive.py`**

`tts_batch/interactive.py`:

```python
"""Guided setup wizard, used when the script is run with no CLI arguments.

Collects answers via input(), builds the equivalent argv list, and hands it
back to cli.main() to parse with the very same argparse parser/validation
used by the direct-CLI path - so there's no separate validation logic to
keep in sync, and invalid answers surface the same parser.error() messages
a mistyped flag would.
"""

from pathlib import Path

from .backends import BACKENDS


def run_wizard(parser) -> list[str]:
    print(
        "No arguments given - starting the guided setup.\n"
        "(Run with --help instead for the full flag reference.)\n"
    )

    default_input = "input.example.txt" if Path("input.example.txt").exists() else None
    prompt = f"Input file [{default_input}]: " if default_input else "Input file: "
    input_path = input(prompt).strip() or default_input
    while not input_path:
        input_path = input("Input file: ").strip()
    argv = [input_path]

    print("\nBackends:")
    names = list(BACKENDS)
    for i, name in enumerate(names, start=1):
        print(f"  [{i}] {name} - {BACKENDS[name].DESCRIPTION}")
    while True:
        choice = input(f"Choose a backend [1-{len(names)}] (default 1): ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            model = names[int(choice) - 1]
            break
        print(f"Enter a number from 1 to {len(names)}.")
    argv += ["--model", model]

    argv += BACKENDS[model].interactive_args({})

    default_output_dir = parser.get_default("output_dir")
    output_dir = input(f"\nOutput directory [{default_output_dir}]: ").strip() or str(
        default_output_dir
    )
    argv += ["--output-dir", output_dir]

    default_gap_ms = parser.get_default("gap_ms")
    gap_ms = input(f"Gap between clips in ms [{default_gap_ms}]: ").strip() or str(default_gap_ms)
    argv += ["--gap-ms", gap_ms]

    print("\nEquivalent command:")
    print("  python generate_and_concat.py " + " ".join(argv))
    print()

    return argv
```

- [ ] **Step 2: Create `cli.py`**

`tts_batch/cli.py`:

```python
"""Argument parsing/validation and the main entry point.

build_parser() assembles the shared flags plus every backend's own
(backend.add_cli_arguments); validate_args() runs shared checks (there are
none today) then delegates to the selected backend's own validate_args.
"""

import argparse
import sys
from pathlib import Path

from . import interactive, runner
from .backends import BACKENDS, kyutai


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TTS clips with a choice of backends and concatenate them per "
            "group. Run with no arguments for a guided setup."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Where to write .wav files"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(BACKENDS),
        default="kyutai",
        help="TTS backend to use (default: kyutai). "
        + " ".join(f"{name}: {backend.DESCRIPTION}." for name, backend in BACKENDS.items()),
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "fr"],
        default="en",
        help=(
            "For --model kyutai: picks a default voice for this language (ignored "
            "if --voice is set). For --model cartesia: passed straight through as "
            "the API's language parameter. Ignored by --model tortoise/breeze "
            "(English-only; --language fr is rejected for those)."
        ),
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help=(
            "Voice to use. For --model kyutai (overrides --language): a path under the "
            "voice repo root WITHOUT the trailing .<hash>@<epoch>.safetensors suffix "
            "(e.g. 'cml-tts/fr/10087_11650_000028-0002.wav') to fetch from Hugging Face, "
            "or a path to a .safetensors file already on disk. For --model tortoise: the "
            "name of a built-in preset voice (e.g. 'tom', 'angie'). For --model cartesia: "
            "a voice_id from your Cartesia voice library (copy one from "
            "play.cartesia.ai). Required for tortoise/cartesia unless --all-voices is set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device to run on (default: cpu; --model breeze always uses CUDA)",
    )
    parser.add_argument(
        "--gap-ms", type=float, default=300.0, help="Pause between clips within a group"
    )
    parser.add_argument(
        "--all-voices",
        action="store_true",
        help=(
            "Generate every group for every voice, instead of just one: every voice "
            "in the voice repo (901+ voices) for --model kyutai, every built-in "
            "preset voice for --model tortoise, or every voice in your account's "
            "library for --model cartesia. Ignores --voice/--language. Writes to "
            "<output-dir>/<group><e if enhanced><voice index>.wav, with a "
            "voices_manifest.txt mapping each index back to its source voice. "
            "Resumable: rerunning the same command skips groups already written."
        ),
    )
    parser.add_argument(
        "--all-fr",
        action="store_true",
        help=(
            f"Like --all-voices, but restricted to voices under '{kyutai.FR_VOICES_PREFIX}' "
            "(the French voices) instead of the whole repo. --model kyutai only."
        ),
    )
    parser.add_argument(
        "--all-eng",
        action="store_true",
        help=(
            f"Like --all-voices, but restricted to voices under '{kyutai.EN_VOICES_PREFIX}' "
            "(the English Expresso voices) instead of the whole repo. --model kyutai only."
        ),
    )
    parser.add_argument(
        "--voice-limit",
        type=int,
        default=None,
        help="With --all-voices, only process the first N voices (for testing).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="With --all-voices, skip the confirmation prompt before a long run.",
    )

    for backend in BACKENDS.values():
        backend.add_cli_arguments(parser)

    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    BACKENDS[args.model].validate_args(args, parser)


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        argv = interactive.run_wizard(parser)
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    validate_args(args, parser)
    runner.run(args)
```

- [ ] **Step 3: Replace `generate_and_concat.py` with the thin entry point**

Replace the entire contents of `generate_and_concat.py` with:

```python
"""Thin entry point - the actual implementation lives in the tts_batch package."""

from tts_batch.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify flag parity against the original script**

Run:

```bash
python -c "
from tts_batch.cli import build_parser
parser = build_parser()
flags = sorted(a.option_strings[0] for a in parser._actions if a.option_strings)
print(flags)
"
```

Expected output (one sorted list, order doesn't matter here — this checks *membership*, not display order):

```
['--all-eng', '--all-fr', '--all-voices', '--breeze-cfg-scale', '--breeze-fast', '--breeze-instruction', '--breeze-model-dir', '--breeze-ref-audio', '--breeze-ref-text', '--breeze-repo-dir', '--breeze-seed', '--cartesia-model', '--device', '--gap-ms', '--help', '--hf-repo', '--language', '--model', '--output-dir', '--tortoise-preset', '--voice', '--voice-limit', '--yes']
```

This is the same 22 flags (plus argparse's own `--help`) as the original `main()` in `generate_and_concat.py` before this refactor — confirms no flag was dropped or misspelled while moving definitions into backend modules. (`--help`'s *grouping* in the rendered `--help` text now clusters backend-specific flags together at the end instead of interleaving them with the shared ones — see the Global Constraints note on this file; that's a deliberate, disclosed cosmetic change, not a dropped/renamed flag.)

- [ ] **Step 5: Verify `--help` runs end-to-end**

Run:

```bash
python generate_and_concat.py --help
```

Expected: argparse's usual `--help` output, listing `input` and all 22 `--flag`s (grouped: shared flags first, then `--hf-repo`, `--tortoise-preset`, the 8 `--breeze-*` flags, `--cartesia-model`), exits 0, no traceback.

- [ ] **Step 6: Commit**

```bash
git add tts_batch/cli.py tts_batch/interactive.py generate_and_concat.py
git commit -m "ADD: cli.py, interactive.py wizard, and reduce generate_and_concat.py to a thin entry point"
```

---

## Task 8: End-to-end verification

**Files:** none (verification only — no code changes expected; if any step below fails, fix the relevant file from Tasks 1-7 and re-run this task's steps)

- [ ] **Step 1: Real end-to-end run with the kyutai backend**

The main `.venv` already has `moshi`/`torch` installed (per `requirements.txt`), so this is a real generation, not just a smoke test. Run in a scratch output directory so it doesn't collide with any existing `output/`:

```bash
python generate_and_concat.py input.example.txt --model kyutai --device cpu --output-dir output_refactor_check --yes
```

Expected: prints `Loading model...`, then a `Wrote output_refactor_check\intro.wav (...)` line and a `Wrote output_refactor_check\chapter_one.wav (...)` line (durations vary), exits 0. Confirm both files exist and are non-empty:

```bash
python -c "from pathlib import Path; p = Path('output_refactor_check'); print(sorted(f.name for f in p.iterdir())); print([f.stat().st_size for f in p.iterdir()])"
```

Expected: `['chapter_one.wav', 'intro.wav']` and two nonzero sizes. Then remove the scratch directory:

```bash
rm -rf output_refactor_check
```

- [ ] **Step 2: Resumable-run behavior still works**

Re-run the same command once more against a fresh scratch dir, kill it after the first file is written isn't practical to script deterministically — instead verify the *skip-existing* logic directly:

```bash
mkdir -p output_refactor_check2
python -c "from pathlib import Path; Path('output_refactor_check2/intro.wav').write_bytes(b'not-real-audio')"
python generate_and_concat.py input.example.txt --model kyutai --device cpu --output-dir output_refactor_check2 --yes
```

Expected: only a `Wrote output_refactor_check2\chapter_one.wav (...)` line — `intro.wav` is skipped because it already exists (the placeholder bytes are left untouched), confirming `audio.generate_groups_for_voice`'s skip-if-exists logic survived the move into `audio.py` unchanged. Then:

```bash
rm -rf output_refactor_check2
```

- [ ] **Step 3: Wizard smoke test via simulated stdin**

Simulate a full guided-setup run for the kyutai backend, single-voice, default language/voice/device, into a scratch output dir:

```bash
printf 'input.example.txt\n1\n1\nen\n\ncpu\noutput_refactor_check3\n300\n' | python generate_and_concat.py
```

Walk through what each line answers: input file, backend choice (`1` = kyutai), kyutai mode (`1` = one voice), language (`en`), voice (blank = default), device (`cpu`), output dir (`output_refactor_check3`), gap-ms (`300`).

Expected: prints the guided-setup banner, the backend menu, `Equivalent command: python generate_and_concat.py input.example.txt --model kyutai --language en --device cpu --output-dir output_refactor_check3 --gap-ms 300`, then actually runs and writes `output_refactor_check3/intro.wav` and `output_refactor_check3/chapter_one.wav`. Confirm and clean up:

```bash
python -c "from pathlib import Path; print(sorted(f.name for f in Path('output_refactor_check3').iterdir()))"
rm -rf output_refactor_check3
```

- [ ] **Step 4: Confirm lazy imports still isolate the optional backends**

```bash
python -c "import tts_batch.cli; print('cli imports cleanly without tortoise-tts/breeze_infer/cartesia installed')"
```

Expected: prints the message, no `ModuleNotFoundError`. (This is the same guarantee checked per-module in Tasks 3-5; this step confirms it also holds once everything is wired together through `cli.py`.)

- [ ] **Step 5: Argument-validation parity spot checks**

Confirm a few `parser.error()` messages are byte-identical to the pre-refactor script (these are user-facing and must not have drifted while moving into backend modules):

```bash
python generate_and_concat.py input.example.txt --model tortoise 2>&1 | tail -1
python generate_and_concat.py input.example.txt --model cartesia 2>&1 | tail -1
```

Expected, respectively:
```
generate_and_concat.py: error: --model tortoise requires --voice <preset-name> (e.g. 'tom', 'angie'), or --all-voices to generate every built-in preset.
```
```
generate_and_concat.py: error: --model cartesia requires --voice <voice_id> (copy one from play.cartesia.ai), or --all-voices to loop over your account's whole voice library.
```

- [ ] **Step 6: Remove stale bytecode cache**

The old `__pycache__/generate_and_concat.cpython-*.pyc` files were compiled from the pre-refactor monolithic script and are now stale:

```bash
rm -rf __pycache__
```

(No commit needed — `__pycache__` is regenerated automatically and is expected to already be gitignored; verify with `git status` that nothing tracked changed from this removal.)

- [ ] **Step 7: Commit any fixes made during this task**

If Steps 1-5 required fixing a bug in Tasks 1-7's files, commit that fix now:

```bash
git status
```

If there are staged/unstaged changes to `tts_batch/` or `generate_and_concat.py` from bug fixes, `git add` them and commit with a message describing the specific bug fixed. If `git status` shows nothing to commit, skip this step.

---

## Task 9: Update README.md and README.fr.md

**Files:**
- Modify: `README.md`
- Modify: `README.fr.md`

- [ ] **Step 1: Add the guided-setup mention to README.md's "Run it" section**

In `README.md`, find:

```markdown
## 3. Run it

```bash
python generate_and_concat.py input.example.txt --output-dir output --gap-ms 300
```
```

Replace with:

```markdown
## 3. Run it

Run with no arguments at all (`python generate_and_concat.py`) for a guided
setup that asks for your input file, backend, and voice, then prints the
equivalent command line before running it — copy that command into a script
for repeat/unattended runs (e.g. under `nohup`). Every `--flag` documented
below still works exactly as before; the wizard is just another way to
build the same command.

```bash
python generate_and_concat.py input.example.txt --output-dir output --gap-ms 300
```
```

- [ ] **Step 2: Add a "Project layout" section to the end of README.md**

Append to the end of `README.md` (after the existing "Using the Cartesia backend" section):

```markdown

## Project layout

The implementation lives in the `tts_batch` package; `generate_and_concat.py`
at the repo root is a thin entry point (`python generate_and_concat.py ...`
still works exactly as before).

- `tts_batch/input_parsing.py` — parses the `# name` / clip-lines input format.
- `tts_batch/audio.py` — concatenation, the resumable per-voice generation
  loop, and the voices-manifest writer, shared by every backend.
- `tts_batch/backends/` — one file per TTS backend (`kyutai.py`,
  `tortoise.py`, `breeze.py`, `cartesia.py`), each owning its own CLI flags,
  argument validation, and generation logic. `base.py` documents the
  interface a new backend needs to implement.
- `tts_batch/cli.py` — assembles the argparse parser from the shared flags
  plus each backend's own, and dispatches validation to the selected
  backend.
- `tts_batch/interactive.py` — the guided setup wizard (see "Run it" above).
- `tts_batch/runner.py` — runs a fully-parsed/validated command.

Adding a fifth backend means creating one new file in `tts_batch/backends/`
implementing the shape documented in `base.py`, and adding it to the
`BACKENDS` dict in `tts_batch/backends/__init__.py` — nothing else needs to
change.
```

- [ ] **Step 3: Mirror both changes in README.fr.md**

In `README.fr.md`, find:

```markdown
## 3. Lancer le script

```bash
python generate_and_concat.py input.example.txt --output-dir output --gap-ms 300
```
```

Replace with:

```markdown
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
```

Then append to the end of `README.fr.md` (after the existing "Utiliser le moteur Cartesia" section):

```markdown

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
```

- [ ] **Step 4: Verify**

```bash
python -c "import pathlib; content = pathlib.Path('README.md').read_text(encoding='utf-8'); assert 'Project layout' in content; assert 'tts_batch' in content; print('README.md OK')"
python -c "import pathlib; content = pathlib.Path('README.fr.md').read_text(encoding='utf-8'); assert 'Structure du projet' in content; assert 'tts_batch' in content; print('README.fr.md OK')"
```

Expected: both print their `OK` line.

- [ ] **Step 5: Commit**

```bash
git add README.md README.fr.md
git commit -m "DOC: document the tts_batch package layout and the guided-setup wizard"
```
