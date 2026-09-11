"""Kokoro-82M backend (https://huggingface.co/hexgrad/Kokoro-82M).

A small (82M param), fast, CPU-friendly TTS model - similar speed class to
Piper (well under a second per short clip on CPU, no GPU required), but a
pure-torch pipeline rather than onnxruntime, with noticeably more natural
output. Apache 2.0 licensed overall, and unlike Piper/Kyutai every one of
its 54 built-in voices is commercial-usable: 5 (4 Japanese, 1 French) are
individually CC BY - attribution required, not a usage restriction - the
rest fall under the repo's plain Apache 2.0 license. See
CC_BY_ATTRIBUTION_VOICES below and
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md. No voice
needs excluding, so there's no --kokoro-commercial-safe flag (unlike
--piper-commercial-safe/--kyutai-commercial-safe).

Voices are identified by a Kokoro voice code (e.g. "af_heart", "ff_siwis"):
the first letter is the language ("a"/"b" English, "f" French, "j"
Japanese, "z" Mandarin, "e" Spanish, "h" Hindi, "i" Italian, "p" Brazilian
Portuguese), the second the voice's gender ("f"/"m"). Voice weights are
downloaded on demand from the hexgrad/Kokoro-82M Hugging Face repo, cached
the same way Piper/Kyutai voice files are.
"""

import functools
from pathlib import Path

import numpy as np
import tqdm
from huggingface_hub import list_repo_files

from .. import audio

NAME = "kokoro"
DESCRIPTION = (
    "Kokoro-82M (English/French/Japanese/Mandarin/Spanish/Hindi/Italian/Portuguese, "
    "fast on CPU, no GPU required, Apache 2.0 - requires the kokoro package, see README)"
)

VOICE_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

EN_LANGUAGE_LETTERS = ("a", "b")  # American + British English
FR_LANGUAGE_LETTERS = ("f",)  # French (single voice: ff_siwis)

DEFAULT_VOICE_BY_LANGUAGE = {
    "en": "af_heart",  # American English, highest-graded voice ("A") per VOICES.md
    "fr": "ff_siwis",  # the only French voice; CC BY (SIWIS dataset), see below
}

# Voices individually licensed CC BY (attribution required) rather than
# covered by the repo's blanket Apache 2.0 license - per VOICES.md on the
# model repo. Still fully commercial-usable, just credit the source dataset
# if you ship audio from these voices.
CC_BY_ATTRIBUTION_VOICES = {
    "jf_gongitsune": "Koniwa dataset (gongitsune story) - CC BY",
    "jf_nezumi": "Koniwa dataset (nezuminoyomeiri story) - CC BY",
    "jf_tebukuro": "Koniwa dataset (tebukurowokaini story) - CC BY",
    "jm_kumo": "Koniwa dataset (kumonoito story) - CC BY",
    "ff_siwis": "SIWIS dataset - CC BY",
}

# Rough per-clip generation time used only to warn before a long
# --all-voices run. CPU figure is measured (~0.7-1s/clip for short English/
# French sentences on this dev machine, after the one-time model load).
# GPU figure is an unverified ballpark - no GPU was available to measure it
# here.
ROUGH_SECONDS_PER_CLIP_CPU = 1.5
ROUGH_SECONDS_PER_CLIP_GPU = 0.3


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--kokoro-speed",
        type=float,
        default=1.0,
        help=(
            "Kokoro speaking-rate multiplier (--model kokoro only, default 1.0): "
            "greater than 1 speaks faster, less than 1 speaks slower."
        ),
    )


def validate_args(args, parser) -> None:
    pass


def list_all_voices(language_letters: tuple[str, ...] | None = None) -> list[str]:
    """Every voice code in the hexgrad/Kokoro-82M repo's voices/ folder,
    optionally restricted to voices whose language-letter prefix is in
    `language_letters` (e.g. ("a", "b") for American+British English)."""
    files = list_repo_files(VOICE_REPO)
    voices = (Path(f).stem for f in files if f.startswith("voices/") and f.endswith(".pt"))
    if language_letters is not None:
        voices = (v for v in voices if v[0] in language_letters)
    return sorted(voices)


def load_pipeline(args, lang_letter: str, model=True):
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_letter, device=args.device, model=model)


def synthesize_clip(pipeline, voice_key: str, speed: float, text: str) -> np.ndarray:
    """Run one line of text through the loaded pipeline and return float32 PCM."""
    chunks = [result.audio.numpy() for result in pipeline(text, voice=voice_key, speed=speed)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def run_single_voice(args, groups) -> None:
    print("Loading model...")
    voice_key = args.voice or DEFAULT_VOICE_BY_LANGUAGE[args.language]
    pipeline = load_pipeline(args, voice_key[0])
    synthesize_fn = functools.partial(synthesize_clip, pipeline, voice_key, args.kokoro_speed)
    audio.generate_groups_for_voice(
        synthesize_fn, SAMPLE_RATE, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    if args.all_fr:
        language_letters = FR_LANGUAGE_LETTERS
    elif args.all_eng:
        language_letters = EN_LANGUAGE_LETTERS
    else:
        language_letters = None
    voice_keys = list_all_voices(language_letters)
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

    print("Loading model...")
    # The underlying model weights are shared across every language/voice -
    # only the (cheap) phonemizer/pipeline differs per language letter - so
    # build one pipeline per distinct letter encountered and reuse the first
    # pipeline's already-loaded model for the rest, instead of reloading the
    # full model for each of the 54 voices.
    pipelines: dict[str, object] = {}
    shared_model = True
    for index, voice_key in enumerate(tqdm.tqdm(voice_keys, desc="voices"), start=1):
        try:
            lang_letter = voice_key[0]
            if lang_letter not in pipelines:
                pipelines[lang_letter] = load_pipeline(args, lang_letter, model=shared_model)
                if shared_model is True:
                    shared_model = pipelines[lang_letter].model
            synthesize_fn = functools.partial(
                synthesize_clip, pipelines[lang_letter], voice_key, args.kokoro_speed
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
            # A single bad/missing voice shouldn't abort a run spanning the
            # whole catalog.
            print(f"Skipping voice {voice_key!r} after error: {e!r}")


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Kokoro options --")
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
            "Kokoro voice code (leave blank for the default voice for the language, "
            "e.g. 'af_heart' - see VOICES.md on the model's Hugging Face repo): "
        ).strip()
        if voice:
            argv += ["--voice", voice]

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
