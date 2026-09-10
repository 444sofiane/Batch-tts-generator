"""Coqui XTTS-v2 backend (https://huggingface.co/coqui/XTTS-v2), via the
community-maintained fork of the (now archived) Coqui TTS toolkit -
https://github.com/idiap/coqui-ai-TTS, installed as the `coqui-tts` package
(the `TTS` import name is unchanged).

Multilingual (17 languages, including English/French) local model that runs
on CPU or GPU. Two ways to pick a voice: one of ~58 built-in "studio
speaker" voices baked into the checkpoint (via --voice), or clone a voice
from a few seconds of reference audio (via --xtts-speaker-wav) - no
separate weights/repo/venv needed beyond installing coqui-tts + torch in
the main .venv, unlike Tortoise.
"""

import functools
import os
from pathlib import Path

import numpy as np
import tqdm

from .. import audio

NAME = "xtts"
DESCRIPTION = (
    "Coqui XTTS-v2 (multilingual incl. English/French, built-in studio voices or "
    "clone from a reference clip - requires the coqui-tts package, see README)"
)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Rough per-clip generation time used only to warn before a long
# --all-voices run. CPU figure is measured (~4.2s/clip average over 5 short
# French sentences on this dev machine, plus ~20s one-time model load - not
# counted here since it's paid once, not per clip); real sentences will
# vary. GPU figure is an unverified ballpark - no GPU was available to
# measure it here.
ROUGH_SECONDS_PER_CLIP_CPU = 6
ROUGH_SECONDS_PER_CLIP_GPU = 1


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--xtts-speaker-wav",
        type=Path,
        default=None,
        help=(
            "Reference audio clip to clone a voice from (--model xtts only), instead "
            "of a built-in speaker name passed via --voice - a few seconds of clean "
            "speech is enough. Mutually exclusive with --voice."
        ),
    )


def validate_args(args, parser) -> None:
    if args.voice and args.xtts_speaker_wav:
        parser.error(
            "--voice and --xtts-speaker-wav are mutually exclusive with --model xtts "
            "(pick a built-in speaker name or a reference clip to clone, not both)."
        )
    if not args.all_voices and not args.voice and not args.xtts_speaker_wav:
        parser.error(
            "--model xtts requires --voice <built-in speaker name> or "
            "--xtts-speaker-wav <reference clip>, or --all-voices to loop over every "
            "built-in speaker."
        )
    if args.all_fr or args.all_eng:
        parser.error(
            "--all-fr/--all-eng are --model kyutai/piper only; use --all-voices for "
            "--model xtts (it loops over the built-in studio speakers instead of a "
            "language-split repo)."
        )
    if args.xtts_speaker_wav and not args.xtts_speaker_wav.is_file():
        parser.error(f"--xtts-speaker-wav {str(args.xtts_speaker_wav)!r} does not exist.")
    if os.environ.get("COQUI_TOS_AGREED") != "1":
        parser.error(
            "--model xtts requires COQUI_TOS_AGREED=1 to be set in your environment. "
            "The model weights are under Coqui's CPML license (free for testing/"
            "evaluation/non-commercial use, a paid license is required for commercial "
            "use - see https://coqui.ai/cpml); setting this variable records that you "
            "agree to those terms, and also avoids an interactive [y/n] prompt on the "
            "first download that would otherwise hang under nohup."
        )


def load_model(args):
    from TTS.api import TTS

    print("Loading model...")
    tts = TTS(MODEL_NAME, progress_bar=False)
    return tts.to(args.device)


def synthesize_clip(tts, speaker: str | None, speaker_wav: str | None, language: str, text: str) -> np.ndarray:
    """Run one line of text through the loaded model and return float32 PCM."""
    wav = tts.tts(text=text, speaker=speaker, speaker_wav=speaker_wav, language=language)
    return np.clip(np.array(wav, dtype=np.float32), -1, 1)


def run_single_voice(args, groups) -> None:
    tts = load_model(args)
    speaker_wav = str(args.xtts_speaker_wav) if args.xtts_speaker_wav else None
    synthesize_fn = functools.partial(synthesize_clip, tts, args.voice, speaker_wav, args.language)
    audio.generate_groups_for_voice(
        synthesize_fn, tts.synthesizer.output_sample_rate, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    tts = load_model(args)
    speakers = sorted(tts.speakers or [])
    if not speakers:
        raise RuntimeError(
            "This XTTS checkpoint reports no built-in speakers to loop over "
            f"({MODEL_NAME} should have ~58; something's off with the download)."
        )
    if args.voice_limit:
        speakers = speakers[: args.voice_limit]

    seconds_per_clip = (
        ROUGH_SECONDS_PER_CLIP_GPU if args.device == "cuda" else ROUGH_SECONDS_PER_CLIP_CPU
    )
    clips_per_voice = sum(len(lines) for _, lines in groups)
    total_clips = clips_per_voice * len(speakers)
    estimate_hours = total_clips * seconds_per_clip / 3600
    message = (
        f"{len(speakers)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Rough estimate at ~{seconds_per_clip}s/clip: {estimate_hours:.1f} hours "
        f"(device={args.device}, very approximate)."
    )
    if not audio.confirm_or_abort(message, args.yes):
        return

    audio.write_voices_manifest(args.output_dir, speakers)

    sample_rate = tts.synthesizer.output_sample_rate
    for index, speaker in enumerate(tqdm.tqdm(speakers, desc="voices"), start=1):
        try:
            synthesize_fn = functools.partial(synthesize_clip, tts, speaker, None, args.language)
            audio.generate_groups_for_voice(
                synthesize_fn,
                sample_rate,
                groups,
                args.output_dir,
                args.gap_ms,
                filename_suffix=str(index),
            )
        except Exception as e:
            # A single bad speaker shouldn't abort a run spanning the whole catalog.
            print(f"Skipping voice {speaker!r} after error: {e!r}")


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- XTTS options --")
    if input("Generate for every built-in studio speaker? [y/N]: ").strip().lower() == "y":
        argv.append("--all-voices")
    else:
        mode = (
            input(
                "Clone from a reference clip, or use a built-in speaker name? "
                "[clip/name] (default name): "
            )
            .strip()
            .lower()
            or "name"
        )
        if mode.startswith("c"):
            speaker_wav = input("Path to a reference audio clip (a few seconds of clean speech): ").strip()
            while not speaker_wav:
                speaker_wav = input("Path to a reference audio clip: ").strip()
            argv += ["--xtts-speaker-wav", speaker_wav]
        else:
            speaker = input(
                "Built-in speaker name (e.g. 'Claribel Dervla' - see README for how to "
                "list them): "
            ).strip()
            while not speaker:
                speaker = input("Built-in speaker name: ").strip()
            argv += ["--voice", speaker]

    language = input("Language, en or fr [en]: ").strip() or "en"
    argv += ["--language", language]

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
