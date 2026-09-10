"""Argument parsing/validation and the main entry point.

build_parser() assembles the shared flags plus every backend's own
(backend.add_cli_arguments); validate_args() runs shared checks (there are
none today) then delegates to the selected backend's own validate_args.
"""

import argparse
import sys
from pathlib import Path

from . import env, interactive, runner
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
        choices=sorted(kyutai.DEFAULT_VOICE_BY_LANGUAGE),
        default="en",
        help=(
            "For --model kyutai/piper: picks a default voice for this language "
            "(ignored if --voice is set). For --model cartesia/xtts: passed straight "
            "through as the API's/model's language parameter. Ignored by --model "
            "tortoise/breeze (English-only; --language fr is rejected for those)."
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
            "play.cartesia.ai). For --model piper: a Piper voice id (e.g. "
            "'en_US-lessac-medium'), overrides --language same as kyutai. For --model "
            "xtts: a built-in studio speaker name (e.g. 'Claribel Dervla'); mutually "
            "exclusive with --xtts-speaker-wav. Required for tortoise/cartesia unless "
            "--all-voices is set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device to run on (default: cpu; --model breeze always uses CUDA). "
            "torch device for kyutai/tortoise/breeze/xtts, onnxruntime CPU-vs-CUDA for piper."
        ),
    )
    parser.add_argument(
        "--gap-ms", type=float, default=300.0, help="Pause between clips within a group"
    )
    parser.add_argument(
        "--all-voices",
        action="store_true",
        help=(
            "Generate every group for every voice, instead of just one: every voice "
            "in the voice repo (901+ voices) for --model kyutai, every voice in the "
            "Piper voice catalog for --model piper, every built-in preset voice for "
            "--model tortoise, every built-in studio speaker (~58) for --model xtts, "
            "or every voice in your account's library for --model cartesia. Ignores "
            "--voice/--language. Writes to "
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
            "(the French voices) instead of the whole repo, for --model kyutai; or to "
            "French voices in the Piper catalog for --model piper."
        ),
    )
    parser.add_argument(
        "--all-eng",
        action="store_true",
        help=(
            f"Like --all-voices, but restricted to voices under '{kyutai.EN_VOICES_PREFIX}' "
            "(the English Expresso voices) instead of the whole repo, for --model kyutai; "
            "or to English voices in the Piper catalog for --model piper."
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
    env.load_dotenv()
    parser = build_parser()
    if len(sys.argv) == 1:
        argv = interactive.run_wizard(parser)
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    validate_args(args, parser)
    runner.run(args)
