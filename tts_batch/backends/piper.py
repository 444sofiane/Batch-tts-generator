"""Piper TTS backend (https://github.com/OHF-Voice/piper1-gpl).

A fast, free, open-weight local model built on onnxruntime rather than
torch: no GPU needed to run well below real-time, which makes it a much
better fit for large-scale batch generation on CPU than Kyutai/Tortoise/
Breeze. The trade-off is quality - it sounds more "robotic" than the other
engines, in exchange for speed and cost.

Voices are identified by a Piper voice id (e.g. "en_US-arctic-medium" or
"fr_FR-siwis-medium") and downloaded on demand from the rhasspy/piper-voices
Hugging Face repo, cached the same way Kyutai's voice files are.
"""

import functools
import json
import os
from pathlib import Path

import numpy as np
import tqdm
from huggingface_hub import hf_hub_download

from .. import audio
from ..translation import Translator

NAME = "piper"

# onnxruntime has no built-in env var for this (checked: its session-creation
# code doesn't read one) - PiperVoice.load() always builds a bare
# SessionOptions() with the library's own default intra-op thread count. Set
# this to override it; leave it unset to keep that default untouched.
INTRA_OP_THREADS_ENV_VAR = "PIPER_INTRA_OP_THREADS"
DESCRIPTION = (
    "Piper TTS (English/French/many other languages, onnxruntime-based - fast on "
    "CPU with no GPU required, best fit for large-scale batches - requires the "
    "piper-tts package, see README)"
)

VOICE_REPO = "rhasspy/piper-voices"
VOICE_CATALOG_FILE = "voices.json"

# Bundled license audit (see ../../PIPER_VOICE_LICENSES.md for the full
# report and methodology) - generated from each voice's MODEL_CARD, with a
# handful cross-checked against the actual source license. Not legal advice;
# --piper-commercial-safe uses this to filter, but verify before relying on
# it for anything you intend to sell.
VOICE_LICENSES_FILE = Path(__file__).parent / "piper_voice_licenses.json"

DEFAULT_VOICE_BY_LANGUAGE = {
    # en_US-arctic-medium (CMU ARCTIC, BSD-style license) rather than the
    # more commonly recommended en_US-lessac-medium: lessac is built on the
    # Blizzard 2013 corpus, whose license explicitly prohibits commercial
    # use - see PIPER_VOICE_LICENSES.md. arctic is verified commercial-safe.
    "en": "en_US-arctic-medium",
    "fr": "fr_FR-siwis-medium",  # CC-BY 4.0, commercial OK with attribution
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
    parser.add_argument(
        "--piper-translate",
        action="store_true",
        help=(
            "Translate input text from French to each Piper voice's language before "
            "synthesis (requires deep-translator and internet access)."
        ),
    )
    parser.add_argument(
        "--piper-translation-cache",
        type=Path,
        default=Path("output/piper_translations.json"),
        help="JSON cache for Piper translations (default: output/piper_translations.json).",
    )
    parser.add_argument(
        "--piper-translation-source",
        default="fr",
        help="Language of the input corpus when --piper-translate is used (default: fr).",
    )
    parser.add_argument(
        "--piper-translation-email",
        default=None,
        help=(
            "Contact email passed to MyMemory (the translation API used by "
            "--piper-translate) to raise its free daily quota from ~5000 to ~50000 "
            "characters - see https://mymemory.translated.net."
        ),
    )
    parser.add_argument(
        "--piper-translate-only",
        action="store_true",
        help=(
            "With --piper-translate: fill the translation cache for every voice's "
            "language and exit, without loading Piper or generating any audio. Lets "
            "you run the slow/flaky network step separately from generation, and "
            "re-run it alone if some translations fail."
        ),
    )
    parser.add_argument(
        "--piper-translate-cached-only",
        action="store_true",
        help=(
            "With --all-voices/--all-fr/--all-eng and --piper-translate: skip voices "
            "whose language doesn't already have every line cached in "
            "--piper-translation-cache, instead of attempting a live translation (which "
            "may fail, e.g. on an exhausted MyMemory quota). Lets you generate what's "
            "already translated now and pick up the rest later - re-run without this flag "
            "once more translations are cached."
        ),
    )
    parser.add_argument(
        "--piper-commercial-safe",
        action="store_true",
        help=(
            "Restrict --all-voices/--all-fr/--all-eng to voices whose license is known "
            "to permit commercial use (CC0/public domain/CC-BY/CC-BY-SA/Apache, or "
            "individually verified) per PIPER_VOICE_LICENSES.md - excludes non-commercial, "
            "copyleft, and unverified voices. With a single --voice, rejects it upfront if "
            "it isn't on that list, instead of generating with it. Not legal advice - "
            "verify before relying on this for anything you intend to sell."
        ),
    )


def validate_args(args, parser) -> None:
    if args.piper_translate_only and not args.piper_translate:
        parser.error("--piper-translate-only requires --piper-translate.")
    if args.piper_translate_cached_only and not args.piper_translate:
        parser.error("--piper-translate-cached-only requires --piper-translate.")
    if args.piper_commercial_safe and args.voice and not is_commercial_safe(args.voice):
        parser.error(
            f"--piper-commercial-safe is set but --voice {args.voice!r} isn't on the "
            "commercial-safe list (or isn't a recognized voice) - see "
            "PIPER_VOICE_LICENSES.md, or drop --piper-commercial-safe if you've "
            "verified its license yourself."
        )


def load_voice_catalog() -> dict:
    """The full rhasspy/piper-voices/voices.json, keyed by voice id.

    Downloaded/cached via huggingface_hub the same way voice files are, so
    it's only fetched over the network once per machine.
    """
    catalog_path = hf_hub_download(VOICE_REPO, VOICE_CATALOG_FILE)
    return json.loads(Path(catalog_path).read_text(encoding="utf-8"))


@functools.lru_cache
def load_voice_licenses() -> dict:
    """The bundled voice-id -> {commercial_safe, license, note, category} audit."""
    return json.loads(VOICE_LICENSES_FILE.read_text(encoding="utf-8"))


def is_commercial_safe(voice_key: str) -> bool:
    """False for voices known to be non-commercial/copyleft/unverified, and for any
    voice added upstream since the audit was last generated (fail conservative)."""
    entry = load_voice_licenses().get(voice_key)
    return bool(entry and entry["commercial_safe"])


def list_all_voices(
    language_family: str | None, catalog: dict, commercial_safe_only: bool = False
) -> list[str]:
    """Every voice id in the catalog, optionally restricted to one language family
    (e.g. 'fr' for every fr_FR/fr_BE/... voice) and/or to commercial-safe voices
    (see is_commercial_safe/PIPER_VOICE_LICENSES.md)."""
    return sorted(
        key
        for key, entry in catalog.items()
        if (language_family is None or entry["language"]["family"] == language_family)
        and (not commercial_safe_only or is_commercial_safe(key))
    )


def download_voice_files(voice_key: str, catalog: dict) -> tuple[Path, Path]:
    """Download (if not already cached) and return local paths to a voice's
    .onnx model and .onnx.json config files."""
    entry = catalog.get(voice_key)
    if entry is None:
        raise ValueError(
            f"Unknown Piper voice {voice_key!r}. Browse "
            "https://rhasspy.github.io/piper-samples for available voice ids "
            "(e.g. 'en_US-arctic-medium', 'fr_FR-siwis-medium')."
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
    tts_voice = PiperVoice.load(onnx_path, config_path=json_path, use_cuda=args.device == "cuda")

    threads = os.environ.get(INTRA_OP_THREADS_ENV_VAR)
    if threads:
        # PiperVoice.load() has no parameter to pass this through, so rebuild
        # just the onnxruntime session with it set, matching load()'s own
        # provider selection (see piper/voice.py) - everything else about the
        # loaded voice (config, espeak data dir, etc.) stays as load() set it.
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = int(threads)
        providers = (
            [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})]
            if args.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        tts_voice.session = onnxruntime.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers
        )

    return tts_voice


def voice_language(voice_key: str, catalog: dict) -> str:
    return catalog[voice_key]["language"]["family"]


def prefill_translations(groups, target_languages: set[str], translator: Translator) -> None:
    """Translate every line into every target language up front and let each
    result land in the cache as it succeeds (Translator.translate saves after
    every new entry - see translation.py).

    Run before the --all-voices generation loop so a transient translation
    failure is reported here, once, instead of silently costing an entire
    voice's worth of clips mid-run (see run_all_voices: a voice whose
    translated_groups() call raises is skipped outright).
    """
    texts = [text for _, lines in groups for text in lines]
    pending = [(language, text) for language in sorted(target_languages) for text in texts]

    failures = []
    for language, text in tqdm.tqdm(pending, desc="pre-translating"):
        try:
            translator.translate(text, language)
        except Exception as e:
            failures.append((language, text, e))

    if failures:
        print(f"{len(failures)}/{len(pending)} translations failed and were left out of the cache:")
        for language, text, e in failures:
            print(f"  [{language}] {text!r}: {e!r}")
        print(
            "Re-run with --piper-translate (--piper-translate-only to skip "
            "generation) to retry just the missing ones - already-cached "
            "translations are reused, not redone."
        )
    else:
        print(f"Pre-translated {len(pending)} (language, line) pairs; cache is warm.")


def language_fully_cached(groups, translator: Translator, target_language: str) -> bool:
    """True if every line across groups already has a cached translation for
    this language - i.e. generating this voice needs no live Argos/MyMemory
    call at all (see --piper-translate-cached-only)."""
    return all(
        translator.is_cached(text, target_language) for _, lines in groups for text in lines
    )


def translated_groups(args, groups, voice_key: str, catalog: dict, translator: Translator | None):
    if translator is None:
        return groups

    target_language = voice_language(voice_key, catalog)
    return [
        (
            group_name,
            [translator.translate(text, target_language) for text in lines],
        )
        for group_name, lines in groups
    ]


def synthesize_clip(tts_voice, syn_config, text: str) -> np.ndarray:
    """Run one line of text through the loaded voice and return float32 PCM."""
    chunks = [chunk.audio_float_array for chunk in tts_voice.synthesize(text, syn_config=syn_config)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def run_single_voice(args, groups) -> None:
    from piper import SynthesisConfig

    catalog = load_voice_catalog()
    voice_key = args.voice or DEFAULT_VOICE_BY_LANGUAGE[args.language]
    translator = (
        Translator(
            args.piper_translation_source,
            args.piper_translation_cache,
            contact_email=args.piper_translation_email,
        )
        if args.piper_translate
        else None
    )
    if translator is not None:
        prefill_translations(groups, {voice_language(voice_key, catalog)}, translator)
        if args.piper_translate_only:
            return
    groups = translated_groups(args, groups, voice_key, catalog, translator)
    print("Loading model...")
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
    voice_keys = list_all_voices(language_family, catalog, args.piper_commercial_safe)
    if args.piper_commercial_safe:
        total_for_scope = len(list_all_voices(language_family, catalog))
        print(
            f"--piper-commercial-safe: {len(voice_keys)}/{total_for_scope} voices in scope "
            "are commercial-safe (see PIPER_VOICE_LICENSES.md); the rest are excluded."
        )
    translator = (
        Translator(
            args.piper_translation_source,
            args.piper_translation_cache,
            contact_email=args.piper_translation_email,
        )
        if args.piper_translate
        else None
    )
    if translator is not None and args.piper_translate_cached_only:
        before = len(voice_keys)
        voice_keys = [
            vk for vk in voice_keys if language_fully_cached(groups, translator, voice_language(vk, catalog))
        ]
        print(
            f"--piper-translate-cached-only: {len(voice_keys)}/{before} voices in scope "
            "already have every line cached; the rest are skipped (not fully translated yet)."
        )

    if args.voice_limit:
        voice_keys = voice_keys[: args.voice_limit]

    if translator is not None:
        target_languages = {voice_language(vk, catalog) for vk in voice_keys}
        prefill_translations(groups, target_languages, translator)
        if args.piper_translate_only:
            return

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
            voice_groups = translated_groups(args, groups, voice_key, catalog, translator)
            audio.generate_groups_for_voice(
                synthesize_fn,
                tts_voice.config.sample_rate,
                voice_groups,
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
            "e.g. 'en_US-arctic-medium' - browse https://rhasspy.github.io/piper-samples): "
        ).strip()
        if voice:
            argv += ["--voice", voice]

    if mode in ("2", "3", "4") and input(
        "Restrict to voices with a known commercial-use-OK license? [y/N] "
        "(see PIPER_VOICE_LICENSES.md): "
    ).strip().lower() == "y":
        argv.append("--piper-commercial-safe")

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
