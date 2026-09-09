"""Breeze-TTS 2 backend (https://huggingface.co/BreezeBlue/Breeze-TTS-2).

Linux + CUDA GPU only, English/Chinese, no voice catalog (clone a reference
clip and/or describe a voice instead). Not pip-installable: point
--breeze-repo-dir at a local clone of github.com/breezeblue-ai/breeze-tts,
whose breeze_infer/models packages are imported directly from that checkout
rather than installed - only once --model breeze is actually used, so this
integration doesn't affect the other backends.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import audio

NAME = "breeze"
DESCRIPTION = (
    "Breeze-TTS 2 (English/Chinese, voice cloning/design, requires a local repo "
    "checkout + weights, Linux + CUDA GPU only - see README)"
)

# Breeze's own infer.py hardcodes these rather than exposing them as CLI
# flags, so they're mirrored here as fixed values to match upstream behavior.
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1


def add_cli_arguments(parser) -> None:
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


def validate_args(args, parser) -> None:
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


def run_single_voice(args, groups) -> None:
    # breeze-tts isn't pip-installable: you clone the repo and its top-level
    # breeze_infer/models packages are imported directly, so the checkout
    # needs to be on sys.path first.
    sys.path.insert(0, str(args.breeze_repo_dir))

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
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=True if args.breeze_fast else None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        repetition_penalty=REPETITION_PENALTY,
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

    audio.generate_groups_for_voice(
        synthesize_fn, runtime.sample_rate, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    raise NotImplementedError(
        "--model breeze has no voice catalog to loop over; validate_args() rejects "
        "--all-voices/--all-fr/--all-eng before this would ever be called."
    )


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Breeze options --")
    repo_dir = input("Path to your breeze-tts repo checkout: ").strip()
    while not repo_dir:
        repo_dir = input("Path to your breeze-tts repo checkout: ").strip()
    argv += ["--breeze-repo-dir", repo_dir]

    model_dir = input("Path to the downloaded Breeze-TTS-2 weights: ").strip()
    while not model_dir:
        model_dir = input("Path to the downloaded Breeze-TTS-2 weights: ").strip()
    argv += ["--breeze-model-dir", model_dir]

    ref_audio = input(
        "Reference voice clip to clone (leave blank to design a voice instead): "
    ).strip()
    if ref_audio:
        ref_text = input("Exact transcript of that reference clip: ").strip()
        while not ref_text:
            ref_text = input("Exact transcript of that reference clip: ").strip()
        argv += ["--breeze-ref-audio", ref_audio, "--breeze-ref-text", ref_text]

    instruction = input(
        "Voice description or delivery instruction [Speak clearly and naturally.]: "
    ).strip()
    if instruction:
        argv += ["--breeze-instruction", instruction]

    if input("Enable fast-path warmup/CUDA graphs? [y/N]: ").strip().lower() == "y":
        argv.append("--breeze-fast")

    return argv
