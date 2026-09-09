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
