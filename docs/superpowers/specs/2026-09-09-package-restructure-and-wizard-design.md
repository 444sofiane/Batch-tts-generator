# Batch TTS generator: package restructure + interactive wizard

Status: approved
Date: 2026-09-09

## Problem

`generate_and_concat.py` is a single 810-line script that implements input
parsing, audio concatenation, CLI argument definition/validation, and four
independent TTS backends (Kyutai, Tortoise, Breeze, Cartesia), each with its
own constants, loading, synthesis, and `--all-voices` looping logic. This
makes it hard to:

- find/change one backend's logic without reading the whole file
- add a new backend without touching a long shared `if/elif` chain in
  `validate_args`/`run()`/`main()`
- use the tool without already knowing the right combination of flags

Goal: split the script into a small package with one file per concern/backend,
and add a guided interactive mode for first-time/occasional use, without
changing behavior for existing CLI/automation usage (e.g. the `nohup ...
--all-fr --yes` pattern documented in the README).

## Non-goals

- No new runtime dependencies (stdlib `input()`/`print()` only for the wizard).
- No behavior change to any existing `--flag` combination — every documented
  command in README.md/README.fr.md must keep working identically.
- No automated test suite — none exists today, and the heavy backends need
  hardware/accounts not available in this dev environment.
- No unification of the three `run_all_voices_*` implementations into one
  generic driver — only the genuinely identical bits (confirm prompt,
  manifest writing) are shared; the per-backend voice-loop logic stays in
  each backend's own file.

## Layout

```
generate_and_concat.py        # thin entry point — unchanged CLI usage
tts_batch/
    __init__.py
    input_parsing.py          # parse_input(), GROUP_HEADER_RE
    audio.py                  # concatenate_clips(), generate_groups_for_voice(),
                               # write_voices_manifest(), confirm_or_abort()
    cli.py                    # build_parser(), validate_args(), main()
    interactive.py            # run_wizard()
    runner.py                 # run(args)
    backends/
        __init__.py           # BACKENDS registry: name -> module
        base.py                # Backend Protocol (typing-only, documents the interface)
        kyutai.py
        tortoise.py
        breeze.py
        cartesia.py
```

`generate_and_concat.py` becomes:

```python
from tts_batch.cli import main

if __name__ == "__main__":
    main()
```

## Backend module interface

Documented as a `typing.Protocol` in `backends/base.py` (duck-typed, not
enforced at runtime — it's there for a coworker adding a new backend to read,
and for IDE/type-checker support). Each of `kyutai.py`, `tortoise.py`,
`breeze.py`, `cartesia.py` implements:

- `NAME: str`
- `add_cli_arguments(parser: argparse.ArgumentParser) -> None` — registers
  this backend's own flags (e.g. `--tortoise-preset`, `--breeze-*`,
  `--cartesia-model`)
- `validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None`
  — raises via `parser.error(...)`, exactly like today's per-backend checks
  in `main()`
- `run_single_voice(args, groups) -> None`
- `run_all_voices(args, groups) -> None` — for `breeze`, this is never
  reachable because `validate_args` rejects `--all-voices` first, but the
  function still exists (raises `NotImplementedError`) so the Protocol holds
  for all four modules
- `interactive_args(common: dict) -> list[str]` — prompts the user (via
  `input()`) for this backend's own settings and returns them as an argv
  fragment, e.g. `["--voice", "tom", "--tortoise-preset", "fast"]`

Backend-owned constants move into their module: `DEFAULT_VOICE_BY_LANGUAGE`,
`FR_VOICES_PREFIX`, `EN_VOICES_PREFIX`, `ROUGH_SECONDS_PER_CLIP_CPU/GPU` →
`kyutai.py`; `TORTOISE_SAMPLE_RATE`, `ROUGH_SECONDS_PER_CLIP_TORTOISE` →
`tortoise.py`; `BREEZE_MAX_NEW_TOKENS`, `BREEZE_MAX_SEQ_LEN`,
`BREEZE_REPETITION_PENALTY` → `breeze.py`; `CARTESIA_SAMPLE_RATE` →
`cartesia.py`.

## cli.py

- `build_parser()` defines the shared/common flags exactly as today
  (`input`, `--output-dir`, `--model`, `--language`, `--voice`, `--device`,
  `--gap-ms`, `--all-voices`, `--all-fr`, `--all-eng`, `--voice-limit`,
  `--yes`), then calls `backend.add_cli_arguments(parser)` for every module
  in `BACKENDS` so backend-specific flags are registered too.
- `validate_args(args, parser)` keeps the model-agnostic checks (if any) and
  then delegates to `BACKENDS[args.model].validate_args(args, parser)` —
  same checks as today's `if args.model == "tortoise": ...` chain in
  `main()`, just dispatched instead of inlined.
- `main()`:
  ```python
  parser = build_parser()
  if len(sys.argv) == 1:
      argv = interactive.run_wizard(parser)
      args = parser.parse_args(argv)
  else:
      args = parser.parse_args()
  validate_args(args, parser)
  runner.run(args)
  ```

## runner.py

```python
def run(args):
    groups = input_parsing.parse_input(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = backends.BACKENDS[args.model]
    if args.all_voices or getattr(args, "all_fr", False) or getattr(args, "all_eng", False):
        backend.run_all_voices(args, groups)
    else:
        backend.run_single_voice(args, groups)
```

This mirrors today's `run()` structure; each backend's own
`run_single_voice`/`run_all_voices` still does its own model loading, same as
now (no shared "load model" step, since loading differs per backend: Kyutai
loads a `TTSModel` up front, Tortoise loads lazily per call, Cartesia has no
local model at all).

## interactive.py — wizard flow

Triggered only when the script runs with **zero CLI arguments**. Any flag
present (including just the positional input file) skips the wizard
entirely, so every existing/scripted/`nohup` invocation is unaffected.

Flow:

1. Input file path (suggest `input.example.txt` if it exists in the cwd).
2. Backend choice — numbered menu built from `BACKENDS`, one line per
   backend (a short `DESCRIPTION` constant per backend module).
3. That backend's own questions via `BACKENDS[choice].interactive_args(common)`.
4. Common options: output dir (default `output`), gap-ms (default `300`).
5. Builds the full argv list, **prints the equivalent command line** (so the
   user can copy it into a script/`nohup` later), then feeds it through the
   same `parser.parse_args()` + `validate_args()` used by the CLI path —
   no separate validation logic for the wizard. If argparse rejects it
   (e.g. a bad number), that error prints and the wizard exits; it does not
   retry/loop.

## Error handling

Unchanged from today except where noted above: all validation stays in
`validate_args` (now dispatched per-backend), all `parser.error()` messages
stay word-for-word the same. The wizard adds no new error paths — it either
produces valid argv (which runs) or invalid argv (which argparse rejects
with its existing message).

## Testing / verification plan

No automated test suite is being added (none exists today; the heavy
backends need hardware/accounts unavailable here). Verification instead:

1. `python generate_and_concat.py --help` output is unchanged from before
   the refactor (same flags, same help text).
2. Real end-to-end smoke test: `python generate_and_concat.py
   input.example.txt --model kyutai --device cpu` (kyutai/moshi is already
   installed in `.venv`) produces the same `.wav` output as before the
   refactor.
3. Wizard smoke test: pipe simulated stdin answers into
   `python generate_and_concat.py` (no args) and confirm it builds the
   expected argv, prints the equivalent command, and runs successfully.
4. `python -c "import tts_batch.cli"` and similar import checks for each
   module, to confirm lazy-imports (tortoise/breeze) aren't accidentally
   promoted to module-level and don't break import when those optional
   packages aren't installed.

## Documentation

`README.md`/`README.fr.md` keep their current structure (already well
organized per-backend); only file-path references that changed (e.g.
pointing at `tts_batch/backends/tortoise.py` instead of
`generate_and_concat.py`) get updated, plus a short new "Project layout"
section explaining the package structure for future coworkers.
