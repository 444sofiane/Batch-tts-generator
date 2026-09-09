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
