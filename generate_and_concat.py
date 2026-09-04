import argparse
import functools
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sphn
import torch
import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

GROUP_HEADER_RE = re.compile(r"^#\s*(.+)$")

# Rough per-clip generation time used only to warn before a long
# --all-voices run; actual time varies a lot with sentence length and the
# specific hardware, this is just a ballpark to size the estimate.
ROUGH_SECONDS_PER_CLIP_CPU = 40
ROUGH_SECONDS_PER_CLIP_GPU = 2

# kyutai/tts-1.6b-en_fr is a single bilingual model; the language of the
# output is just whatever language the input text is in. What actually
# differs per language is the voice recording, so --language only picks a
# default voice — pass --voice directly to use any other one.
DEFAULT_VOICE_BY_LANGUAGE = {
    "en": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "fr": "cml-tts/fr/10087_11650_000028-0002.wav",
}

FR_VOICES_PREFIX = "cml-tts/fr/"
EN_VOICES_PREFIX = "expresso/"

# Tortoise's output is always 24kHz, regardless of voice or preset.
TORTOISE_SAMPLE_RATE = 24000

# Rough per-clip generation time on GPU for each Tortoise quality preset,
# used only to size the --all-voices time estimate (CPU is impractically
# slow for Tortoise and isn't estimated here).
ROUGH_SECONDS_PER_CLIP_TORTOISE = {
    "ultra_fast": 15,
    "fast": 30,
    "standard": 90,
    "high_quality": 240,
}

# Breeze's own infer.py hardcodes these rather than exposing them as CLI
# flags, so they're mirrored here as fixed values to match upstream behavior.
BREEZE_MAX_NEW_TOKENS = 1500
BREEZE_MAX_SEQ_LEN = 2048
BREEZE_REPETITION_PENALTY = 1.1


def parse_input(path: Path) -> list[tuple[str, list[str]]]:
    """Split the input file into (group_name, [clip_text, ...]) blocks.

    A line starting with '#' names the group that follows. Consecutive
    non-blank lines after it are the clips, synthesized and concatenated
    in order. A blank line ends the current group.
    """
    groups: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_name is not None and current_lines:
            groups.append((current_name, current_lines.copy()))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            current_lines.clear()
            current_name = None
            continue

        header = GROUP_HEADER_RE.match(line)
        if header:
            flush()
            current_lines.clear()
            current_name = header.group(1).strip()
        elif current_name is not None:
            current_lines.append(line)
        else:
            raise ValueError(
                f"Line {raw_line!r} has no group header (expected a '# name' line first)"
            )

    flush()
    return groups


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


def concatenate_clips(clips: list[np.ndarray], sample_rate: int, gap_ms: float) -> np.ndarray:
    """Join a group's synthesized clips with a fixed silence gap between them."""
    if not clips:
        return np.zeros(0, dtype=np.float32)

    gap_samples = int(sample_rate * gap_ms / 1000)
    silence = np.zeros(gap_samples, dtype=np.float32)

    pieces = [clips[0]]
    for clip in clips[1:]:
        pieces.append(silence)
        pieces.append(clip)

    return np.concatenate(pieces)


def generate_groups_for_voice(
    synthesize_fn: Callable[[str], np.ndarray],
    sample_rate: int,
    groups: list[tuple[str, list[str]]],
    output_dir: Path,
    gap_ms: float,
    filename_suffix: str = "",
) -> None:
    """Generate+concatenate every group for one voice, skipping already-done groups.

    Backend-agnostic: `synthesize_fn` does the actual per-line generation, so
    this same resumable/skip logic is shared by both the Kyutai and Tortoise
    backends (see run_single_voice_kyutai/run_single_voice_tortoise etc.).

    Skipping existing output files (rather than always overwriting) is what
    makes an interrupted --all-voices run resumable: rerunning the same
    command only fills in what's missing instead of starting over.

    `filename_suffix` disambiguates multiple voices writing into the same
    flat `output_dir` (see run()'s --all-voices/--all-fr handling); it's
    empty for the single-voice case, where <group>.wav alone is unambiguous.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for group_name, lines in tqdm.tqdm(groups, desc="groups", leave=False):
        out_path = output_dir / f"{group_name}{filename_suffix}.wav"
        if out_path.exists():
            continue

        clips = [
            synthesize_fn(text)
            for text in tqdm.tqdm(lines, desc=group_name, leave=False)
        ]
        audio = concatenate_clips(clips, sample_rate, gap_ms)
        sphn.write_wav(str(out_path), audio, sample_rate)
        print(f"Wrote {out_path} ({len(audio) / sample_rate:.1f}s)")


def run_single_voice_kyutai(args: argparse.Namespace, tts_model: TTSModel, groups) -> None:
    voice = args.voice or DEFAULT_VOICE_BY_LANGUAGE[args.language]
    voice_path = voice if voice.endswith(".safetensors") else tts_model.get_voice_path(voice)
    condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
    synthesize_fn = functools.partial(synthesize_clip, tts_model, condition_attributes)
    generate_groups_for_voice(
        synthesize_fn, tts_model.mimi.sample_rate, groups, args.output_dir, args.gap_ms
    )


def load_tortoise_tts(device: str):
    """Lazily construct the Tortoise TextToSpeech model.

    Imported here rather than at module scope so that installing
    tortoise-tts (a separate, heavier dependency - see README) is only
    required when --model tortoise is actually used.
    """
    from tortoise.api import TextToSpeech

    return TextToSpeech(device=device)


def synthesize_clip_tortoise(
    tts, voice_samples, conditioning_latents, preset: str, text: str
) -> np.ndarray:
    """Run one line of text through a loaded Tortoise model and return float32 PCM."""
    with torch.no_grad():
        gen = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset,
        )
    return gen.cpu().numpy()[0, 0].astype(np.float32)


def list_all_tortoise_voices() -> list[str]:
    """Every built-in Tortoise preset voice name."""
    from tortoise.utils.audio import get_voices

    # "random" is a pseudo-voice (a random blend, non-reproducible) handled
    # by Tortoise's own CLI rather than a real bundled voice directory; it
    # isn't expected to show up here, but is excluded defensively.
    return sorted(name for name in get_voices() if name != "random")


def run_single_voice_tortoise(args: argparse.Namespace, groups) -> None:
    from tortoise.utils.audio import load_voice

    print("Loading model...")
    tts = load_tortoise_tts(args.device)
    voice_samples, conditioning_latents = load_voice(args.voice)
    synthesize_fn = functools.partial(
        synthesize_clip_tortoise, tts, voice_samples, conditioning_latents, args.tortoise_preset
    )
    generate_groups_for_voice(
        synthesize_fn, TORTOISE_SAMPLE_RATE, groups, args.output_dir, args.gap_ms
    )


def run_single_voice_breeze(args: argparse.Namespace, groups) -> None:
    # breeze-tts isn't pip-installable: you clone the repo and its top-level
    # breeze_infer/models packages are imported directly, so the checkout
    # needs to be on sys.path first.
    sys.path.insert(0, str(args.breeze_repo_dir))
    from dataclasses import replace

    from breeze_infer.runtime import (
        load_runtime,
        resolve_device,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import get_template, prepare_inputs
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    from models.warmup_profile import load_warmup_profile

    print("Loading model...")
    tokenizer, model, audio_tokenizer = load_runtime(
        args.breeze_model_dir, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=BREEZE_MAX_NEW_TOKENS,
        max_seq_len=BREEZE_MAX_SEQ_LEN,
        fast_all=True if args.breeze_fast else None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        repetition_penalty=BREEZE_REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)

    if runtime.fast_enabled:
        fast_config_path = args.breeze_repo_dir / "configs" / "fast.json"
        profile = load_warmup_profile(fast_config_path)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms")

    # Built once and reused for every clip: which template (plain
    # voice-design vs. reference-clone/direction) and which ref audio/text,
    # if any, don't change line to line, only the text does.
    request_base = {"instruction": args.breeze_instruction, "speaker": "S0"}
    if args.breeze_ref_audio is not None:
        request_base["ref_audio_path"] = str(args.breeze_ref_audio)
        request_base["ref_text"] = args.breeze_ref_text.strip()
        template = get_template("ref_edit_tata")
    else:
        template = get_template("tts_instruction")

    def synthesize_fn(text: str) -> np.ndarray:
        request = {**request_base, "id": "clip", "text": text}
        set_all_seeds(args.breeze_seed)
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            template,
            guidance_scale=args.breeze_cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        chunks = [
            chunk.audio
            for chunk in runtime.iter_audio_chunks(inputs, request_id="clip", seed=args.breeze_seed)
        ]
        return np.concatenate(chunks).astype(np.float32)

    generate_groups_for_voice(
        synthesize_fn, runtime.sample_rate, groups, args.output_dir, args.gap_ms
    )


def write_voices_manifest(output_dir: Path, voice_files: list[str]) -> None:
    """Index -> source voice path, since flat multi-voice filenames (e.g.
    'intro47.wav') can't carry that information themselves."""
    manifest_path = output_dir / "voices_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for index, voice_file in enumerate(voice_files, start=1):
            manifest.write(f"{index}\t{voice_file}\n")
    print(f"Wrote voice index -> source mapping to {manifest_path}")


def run_all_voices_kyutai(args: argparse.Namespace, tts_model: TTSModel, groups) -> None:
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
    print(
        f"{len(voice_files)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Rough estimate at ~{seconds_per_clip}s/clip: {estimate_hours:.1f} hours "
        f"(device={args.device}, very approximate)."
    )
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    # All voices write into the same flat output_dir (group.wav alone would
    # collide across voices), so filenames get a sequential per-voice index
    # instead, and a manifest records what each index actually maps to.
    write_voices_manifest(args.output_dir, voice_files)

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
            generate_groups_for_voice(
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


def run_all_voices_tortoise(args: argparse.Namespace, groups) -> None:
    from tortoise.utils.audio import load_voice

    voice_names = list_all_tortoise_voices()
    if args.voice_limit:
        voice_names = voice_names[: args.voice_limit]

    seconds_per_clip = ROUGH_SECONDS_PER_CLIP_TORTOISE[args.tortoise_preset]
    clips_per_voice = sum(len(lines) for _, lines in groups)
    total_clips = clips_per_voice * len(voice_names)
    estimate_hours = total_clips * seconds_per_clip / 3600
    print(
        f"{len(voice_names)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Rough estimate at ~{seconds_per_clip}s/clip ('{args.tortoise_preset}' preset): "
        f"{estimate_hours:.1f} hours (device={args.device}, very approximate)."
    )
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    write_voices_manifest(args.output_dir, voice_names)

    print("Loading model...")
    tts = load_tortoise_tts(args.device)
    for index, voice_name in enumerate(tqdm.tqdm(voice_names, desc="voices"), start=1):
        try:
            voice_samples, conditioning_latents = load_voice(voice_name)
            synthesize_fn = functools.partial(
                synthesize_clip_tortoise,
                tts,
                voice_samples,
                conditioning_latents,
                args.tortoise_preset,
            )
            generate_groups_for_voice(
                synthesize_fn,
                TORTOISE_SAMPLE_RATE,
                groups,
                args.output_dir,
                args.gap_ms,
                filename_suffix=str(index),
            )
        except Exception as e:
            print(f"Skipping voice {voice_name!r} after error: {e!r}")


def run(args: argparse.Namespace) -> None:
    groups = parse_input(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "tortoise":
        if args.all_voices:
            run_all_voices_tortoise(args, groups)
        else:
            run_single_voice_tortoise(args, groups)
        return

    if args.model == "breeze":
        run_single_voice_breeze(args, groups)
        return

    print("Loading model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=32, temp=0.6, device=args.device
    )

    if args.all_voices or args.all_fr or args.all_eng:
        run_all_voices_kyutai(args, tts_model, groups)
    else:
        run_single_voice_kyutai(args, tts_model, groups)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS clips with Kyutai's PyTorch TTS and concatenate them per group."
    )
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Where to write .wav files"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["kyutai", "tortoise", "breeze"],
        default="kyutai",
        help=(
            "TTS backend to use (default: kyutai). tortoise requires the separate "
            "tortoise-tts package (see README) and is English-only. breeze requires "
            "a local breeze-tts checkout + downloaded weights (see README), a CUDA "
            "GPU, and supports English/Chinese, not French."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_DSM_TTS_REPO,
        help="HF repo for the TTS model (--model kyutai only)",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=sorted(DEFAULT_VOICE_BY_LANGUAGE),
        default="en",
        help="Picks a default voice for this language (ignored if --voice is set; --model kyutai only)",
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
            "name of a built-in preset voice (e.g. 'tom', 'angie') - required unless "
            "--all-voices is set."
        ),
    )
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
    parser.add_argument(
        "--breeze-repo-dir",
        type=Path,
        default=None,
        help=(
            "Path to a local clone of github.com/breezeblue-ai/breeze-tts (required "
            "for --model breeze; its breeze_infer/models packages are imported "
            "directly rather than pip-installed)."
        ),
    )
    parser.add_argument(
        "--breeze-model-dir",
        type=Path,
        default=None,
        help=(
            "Path to the downloaded BreezeBlue/Breeze-TTS-2 weights (required for "
            "--model breeze)."
        ),
    )
    parser.add_argument(
        "--breeze-ref-audio",
        type=Path,
        default=None,
        help=(
            "Reference voice clip to clone (--model breeze only). Must be given "
            "together with --breeze-ref-text. Omit both for voice-design mode, "
            "where the voice comes purely from --breeze-instruction."
        ),
    )
    parser.add_argument(
        "--breeze-ref-text",
        type=str,
        default=None,
        help="Exact transcript of --breeze-ref-audio (--model breeze only).",
    )
    parser.add_argument(
        "--breeze-instruction",
        type=str,
        default="Speak clearly and naturally.",
        help=(
            "Voice description (without --breeze-ref-audio) or delivery instruction "
            "(with --breeze-ref-audio) for --model breeze. Default matches Breeze's "
            "own infer.py default."
        ),
    )
    parser.add_argument(
        "--breeze-cfg-scale",
        type=float,
        default=1.0,
        help=(
            "Guidance scale for --model breeze (default: 1.0, matching infer.py's "
            "own default - Breeze's own usage examples use 4 for voice design/"
            "direction)."
        ),
    )
    parser.add_argument(
        "--breeze-seed",
        type=int,
        default=42,
        help="Random seed for --model breeze (default: 42, matching infer.py).",
    )
    parser.add_argument(
        "--breeze-fast",
        action="store_true",
        help=(
            "Enable Breeze's fast-path optimizations (warmup + CUDA graphs) for "
            "--model breeze, equivalent to infer.py's --fast-all."
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
            "in the voice repo (901+ voices) for --model kyutai, or every built-in "
            "preset voice for --model tortoise. Ignores --voice/--language. Writes to "
            "<output-dir>/<group><e if enhanced><voice index>.wav, with a "
            "voices_manifest.txt mapping each index back to its source voice. "
            "Resumable: rerunning the same command skips groups already written."
        ),
    )
    parser.add_argument(
        "--all-fr",
        action="store_true",
        help=(
            f"Like --all-voices, but restricted to voices under '{FR_VOICES_PREFIX}' "
            "(the French voices) instead of the whole repo. --model kyutai only."
        ),
    )
    parser.add_argument(
        "--all-eng",
        action="store_true",
        help=(
            f"Like --all-voices, but restricted to voices under '{EN_VOICES_PREFIX}' "
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
    args = parser.parse_args()

    if args.model == "tortoise":
        if args.all_fr or args.all_eng:
            parser.error(
                "--all-fr/--all-eng are --model kyutai only (Tortoise voices aren't "
                "split by language); use --all-voices instead."
            )
        if args.language == "fr":
            parser.error("Tortoise-TTS is English-only; --language fr requires --model kyutai.")
        if not args.all_voices and not args.voice:
            parser.error(
                "--model tortoise requires --voice <preset-name> (e.g. 'tom', 'angie'), "
                "or --all-voices to generate every built-in preset."
            )
    elif args.model == "breeze":
        if args.breeze_repo_dir is None or args.breeze_model_dir is None:
            parser.error("--model breeze requires --breeze-repo-dir and --breeze-model-dir.")
        if bool(args.breeze_ref_audio) != bool(args.breeze_ref_text):
            parser.error("--breeze-ref-audio and --breeze-ref-text must be given together.")
        if args.voice or args.all_voices or args.all_fr or args.all_eng:
            parser.error(
                "--voice/--all-voices/--all-fr/--all-eng aren't supported with "
                "--model breeze (there's no voice catalog to select from) - use "
                "--breeze-ref-audio/--breeze-ref-text and/or --breeze-instruction "
                "instead."
            )
        if args.language == "fr":
            parser.error(
                "Breeze-TTS supports English/Chinese, not French; --language fr "
                "requires --model kyutai."
            )

    run(args)


if __name__ == "__main__":
    main()
