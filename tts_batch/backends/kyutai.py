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
    "en": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "fr": "cml-tts/fr/10087_11650_000028-0002.wav",
}

FR_VOICES_PREFIX = "cml-tts/fr/"
EN_VOICES_PREFIX = "expresso/"

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


def validate_args(args, parser) -> None:
    pass


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


def list_all_voices(voice_repo: str, prefix: str | None = None) -> list[str]:
    """Every voice embedding file in the voice repo, as full repo-relative paths.

    If `prefix` is given, only files under that path prefix are returned
    (e.g. "cml-tts/fr/" for just the French voices).
    """
    files = list_repo_files(voice_repo)
    voices = (f for f in files if f.endswith(".safetensors"))
    if prefix is not None:
        voices = (f for f in voices if f.startswith(prefix))
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
    voice_files = list_all_voices(tts_model.voice_repo, prefix=voice_prefix)
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

    device = input("Device, cpu or cuda [cpu]: ").strip() or "cpu"
    argv += ["--device", device]
    return argv
