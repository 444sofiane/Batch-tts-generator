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
            "--all-fr/--all-eng are --model kyutai/piper only (Tortoise voices aren't "
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
