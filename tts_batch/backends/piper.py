"""Piper TTS backend (https://github.com/OHF-Voice/piper1-gpl).

A fast, free, open-weight local model built on onnxruntime rather than
torch: no GPU needed to run well below real-time, which makes it a much
better fit for large-scale batch generation on CPU than Kyutai/Tortoise/
Breeze. The trade-off is quality - it sounds more "robotic" than the other
engines, in exchange for speed and cost.

Voices are identified by a Piper voice id (e.g. "en_US-lessac-medium" or
"fr_FR-siwis-medium") and downloaded on demand from the rhasspy/piper-voices
Hugging Face repo, cached the same way Kyutai's voice files are.
"""

import functools
import json
from pathlib import Path

import numpy as np
import tqdm
from huggingface_hub import hf_hub_download

from .. import audio

NAME = "piper"
DESCRIPTION = (
    "Piper TTS (English/French/many other languages, onnxruntime-based - fast on "
    "CPU with no GPU required, best fit for large-scale batches - requires the "
    "piper-tts package, see README)"
)

VOICE_REPO = "rhasspy/piper-voices"
VOICE_CATALOG_FILE = "voices.json"

DEFAULT_VOICE_BY_LANGUAGE = {
    "en": "en_US-lessac-medium",
    "fr": "fr_FR-siwis-medium",
}

FR_LANGUAGE_FAMILY = "fr"
EN_LANGUAGE_FAMILY = "en"

# Rough per-clip generation time used only to warn before a long
# --all-voices run. CPU figure is measured (~0.18s/clip average over 5 short
# French sentences on this dev machine); real sentences will vary, hence
# rounding up. GPU figure is an unverified ballpark - no GPU was available
# to measure it here.
ROUGH_SECONDS_PER_CLIP_CPU = 1
ROUGH_SECONDS_PER_CLIP_GPU = 0.5


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--piper-length-scale",
        type=float,
        default=1.0,
        help=(
            "Piper speaking-rate multiplier (--model piper only, default 1.0): "
            "greater than 1 speaks slower, less than 1 speaks faster."
        ),
    )


def validate_args(args, parser) -> None:
    pass


def load_voice_catalog() -> dict:
    """The full rhasspy/piper-voices/voices.json, keyed by voice id.

    Downloaded/cached via huggingface_hub the same way voice files are, so
    it's only fetched over the network once per machine.
    """
    catalog_path = hf_hub_download(VOICE_REPO, VOICE_CATALOG_FILE)
    return json.loads(Path(catalog_path).read_text(encoding="utf-8"))


def list_all_voices(language_family: str | None, catalog: dict) -> list[str]:
    """Every voice id in the catalog, optionally restricted to one language family
    (e.g. 'fr' for every fr_FR/fr_BE/... voice)."""
    return sorted(
        key
        for key, entry in catalog.items()
        if language_family is None or entry["language"]["family"] == language_family
    )


def download_voice_files(voice_key: str, catalog: dict) -> tuple[Path, Path]:
    """Download (if not already cached) and return local paths to a voice's
    .onnx model and .onnx.json config files."""
    entry = catalog.get(voice_key)
    if entry is None:
        raise ValueError(
            f"Unknown Piper voice {voice_key!r}. Browse "
            "https://rhasspy.github.io/piper-samples for available voice ids "
            "(e.g. 'en_US-lessac-medium', 'fr_FR-siwis-medium')."
        )
    repo_paths = list(entry["files"])
    onnx_repo_path = next(p for p in repo_paths if p.endswith(".onnx"))
    json_repo_path = next(p for p in repo_paths if p.endswith(".onnx.json"))
    onnx_path = hf_hub_download(VOICE_REPO, onnx_repo_path)
    json_path = hf_hub_download(VOICE_REPO, json_repo_path)
    return Path(onnx_path), Path(json_path)


def load_voice(args, voice_key: str, catalog: dict):
    from piper import PiperVoice

    onnx_path, json_path = download_voice_files(voice_key, catalog)
    return PiperVoice.load(onnx_path, config_path=json_path, use_cuda=args.device == "cuda")


def synthesize_clip(tts_voice, syn_config, text: str) -> np.ndarray:
    """Run one line of text through the loaded voice and return float32 PCM."""
    chunks = [chunk.audio_float_array for chunk in tts_voice.synthesize(text, syn_config=syn_config)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def run_single_voice(args, groups) -> None:
    from piper import SynthesisConfig

    print("Loading model...")
    catalog = load_voice_catalog()
    voice_key = args.voice or DEFAULT_VOICE_BY_LANGUAGE[args.language]
    tts_voice = load_voice(args, voice_key, catalog)
    syn_config = SynthesisConfig(length_scale=args.piper_length_scale)
    synthesize_fn = functools.partial(synthesize_clip, tts_voice, syn_config)
    audio.generate_groups_for_voice(
        synthesize_fn, tts_voice.config.sample_rate, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    from piper import SynthesisConfig

    catalog = load_voice_catalog()
    if args.all_fr:
        language_family = FR_LANGUAGE_FAMILY
    elif args.all_eng:
        language_family = EN_LANGUAGE_FAMILY
    else:
        language_family = None
    voice_keys = list_all_voices(language_family, catalog)
    if args.voice_limit:
        voice_keys = voice_keys[: args.voice_limit]

    seconds_per_clip = (
        ROUGH_SECONDS_PER_CLIP_CPU if args.device == "cpu" else ROUGH_SECONDS_PER_CLIP_GPU
    )
    clips_per_voice = sum(len(lines) for _, lines in groups)
    total_clips = clips_per_voice * len(voice_keys)
    estimate_hours = total_clips * seconds_per_clip / 3600
    message = (
        f"{len(voice_keys)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Rough estimate at ~{seconds_per_clip}s/clip: {estimate_hours:.1f} hours "
        f"(device={args.device}, very approximate)."
    )
    if not audio.confirm_or_abort(message, args.yes):
        return

    audio.write_voices_manifest(args.output_dir, voice_keys)

    syn_config = SynthesisConfig(length_scale=args.piper_length_scale)
    for index, voice_key in enumerate(tqdm.tqdm(voice_keys, desc="voices"), start=1):
        try:
            tts_voice = load_voice(args, voice_key, catalog)
            synthesize_fn = functools.partial(synthesize_clip, tts_voice, syn_config)
            audio.generate_groups_for_voice(
                synthesize_fn,
                tts_voice.config.sample_rate,
                groups,
                args.output_dir,
                args.gap_ms,
                filename_suffix=str(index),
            )
        except Exception as e:
            # A single bad/missing voice shouldn't abort a run spanning the
            # whole catalog.
            print(f"Skipping voice {voice_key!r} after error: {e!r}")


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Piper options --")
    print("  [1] One voice (default)")
    print("  [2] All voices in the catalog")
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
            "Piper voice id (leave blank for the default voice for the language, "
            "e.g. 'en_US-lessac-medium' - browse https://rhasspy.github.io/piper-samples): "
        ).strip()
        if voice:
            argv += ["--voice", voice]

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
