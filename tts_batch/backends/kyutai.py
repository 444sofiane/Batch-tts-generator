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
    # unmute-prod-website/default_voice.wav (Kyutai's own recording, CC0)
    # rather than the more commonly seen expresso/ voice: expresso is
    # licensed CC BY-NC 4.0, non-commercial only - see KYUTAI_VOICE_LICENSES.md.
    "en": "unmute-prod-website/default_voice.wav",
    "fr": "cml-tts/fr/10087_11650_000028-0002.wav",  # CC BY 4.0, commercial OK with attribution
}

FR_VOICES_PREFIX = "cml-tts/fr/"
# A tuple, not just "expresso/": vctk/ and ears/ are English too (confirmed
# against their own dataset pages), and str.startswith() accepts a tuple of
# prefixes natively. Without this, --all-eng --kyutai-commercial-safe would
# always yield zero voices, since expresso/ alone is entirely non-commercial.
EN_VOICES_PREFIX = ("expresso/", "vctk/", "ears/")

# Commercial-use classification per voice-repo path prefix, derived from
# kyutai/tts-voices' own README.md (see ../../KYUTAI_VOICE_LICENSES.md for
# the full audit, sources, and methodology). Checked in order - list exact
# single-file entries before the prefix rule for their containing folder.
# True/False = verified safe/unsafe; None = unclear, treated as unsafe by
# is_commercial_safe() (fail conservative). Matches both a bare voice path
# (e.g. "cml-tts/fr/....wav") and its full .safetensors form via prefix, and
# works for any --voice value someone passes since both are checked the
# same way.
VOICE_LICENSE_RULES: list[tuple[str, bool | None, str]] = [
    ("unmute-prod-website/degaulle-2.wav", None,
     "Unclear - probably public domain (1940 historical recording; Kyutai's own README "
     "hedges on the exact license)."),
    ("unmute-prod-website/freesound/", None,
     "Unclear - sourced from freesound.org, whose per-upload license isn't verified here "
     "despite the repo README's blanket CC0 claim."),
    ("unmute-prod-website/ex04_narration_longform_00001.wav", False,
     "CC BY-NC 4.0 (Expresso-derived) - non-commercial only."),
    ("unmute-prod-website/p329_022.wav", True, "CC BY 4.0 (VCTK-derived)."),
    ("unmute-prod-website/", True, "CC0 (Kyutai's own recording)."),
    ("voice-donations/", True, "CC0 (Unmute Voice Donation Project)."),
    ("vctk/", True, "CC BY 4.0 (Voice Cloning Toolkit dataset)."),
    ("cml-tts/fr/", True, "CC BY 4.0 (CML-TTS Dataset)."),
    ("alba-mackenna/", True, "CC BY 4.0."),
    ("expresso/", False, "CC BY-NC 4.0 (Expresso dataset) - non-commercial only."),
    ("ears/", False, "CC BY-NC 4.0 (EARS dataset) - non-commercial only."),
]


def voice_commercial_status(voice_key: str) -> tuple[bool | None, str]:
    """(is_commercial_safe, note) for a voice's repo-relative path, per
    VOICE_LICENSE_RULES. None means unclear/unrecognized."""
    for prefix, safe, note in VOICE_LICENSE_RULES:
        if voice_key.startswith(prefix):
            return safe, note
    return None, "Not in the license audit (unrecognized voice) - see KYUTAI_VOICE_LICENSES.md."


def is_commercial_safe(voice_key: str) -> bool:
    safe, _ = voice_commercial_status(voice_key)
    return bool(safe)


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
    parser.add_argument(
        "--kyutai-commercial-safe",
        action="store_true",
        help=(
            "Restrict --all-voices/--all-fr/--all-eng to voices whose license is known "
            "to permit commercial use (CC0/CC-BY per dataset) per "
            "KYUTAI_VOICE_LICENSES.md - excludes the non-commercial expresso/ and ears/ "
            "datasets and a few unverified voices. With a single --voice, rejects it "
            "upfront if it isn't commercial-safe, instead of generating with it. Not "
            "legal advice - verify before relying on this for anything you intend to sell."
        ),
    )


def validate_args(args, parser) -> None:
    if args.kyutai_commercial_safe and args.voice and not is_commercial_safe(args.voice):
        parser.error(
            f"--kyutai-commercial-safe is set but --voice {args.voice!r} isn't on the "
            "commercial-safe list (or isn't a recognized voice) - see "
            "KYUTAI_VOICE_LICENSES.md, or drop --kyutai-commercial-safe if you've "
            "verified its license yourself."
        )


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


def list_all_voices(
    voice_repo: str, prefix: str | None = None, commercial_safe_only: bool = False
) -> list[str]:
    """Every voice embedding file in the voice repo, as full repo-relative paths.

    If `prefix` is given, only files under that path prefix are returned
    (e.g. "cml-tts/fr/" for just the French voices). If `commercial_safe_only`
    is set, voices not verified commercial-safe are excluded (see
    is_commercial_safe/KYUTAI_VOICE_LICENSES.md).
    """
    files = list_repo_files(voice_repo)
    voices = (f for f in files if f.endswith(".safetensors"))
    if prefix is not None:
        voices = (f for f in voices if f.startswith(prefix))
    if commercial_safe_only:
        voices = (f for f in voices if is_commercial_safe(f))
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
    voice_files = list_all_voices(
        tts_model.voice_repo, prefix=voice_prefix, commercial_safe_only=args.kyutai_commercial_safe
    )
    if args.kyutai_commercial_safe:
        total_for_scope = len(list_all_voices(tts_model.voice_repo, prefix=voice_prefix))
        print(
            f"--kyutai-commercial-safe: {len(voice_files)}/{total_for_scope} voices in scope "
            "are commercial-safe (see KYUTAI_VOICE_LICENSES.md); the rest are excluded."
        )
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

    if mode in ("2", "3", "4") and input(
        "Restrict to voices with a known commercial-use-OK license? [y/N] "
        "(see KYUTAI_VOICE_LICENSES.md): "
    ).strip().lower() == "y":
        argv.append("--kyutai-commercial-safe")

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
