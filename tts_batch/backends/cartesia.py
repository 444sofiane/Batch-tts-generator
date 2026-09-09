"""Cartesia cloud TTS backend (https://play.cartesia.ai/text-to-speech).

Unlike the other backends, this is a cloud API, not a local model: no
download, no GPU, but it does need a CARTESIA_API_KEY and calls Cartesia's
servers for every clip. The `cartesia` package is imported lazily so it's
only required when --model cartesia is actually used.
"""

import functools
import os

import numpy as np
import tqdm

from .. import audio

NAME = "cartesia"
DESCRIPTION = (
    "Cartesia's cloud API (English/French, no local model or GPU, requires the "
    "cartesia package and CARTESIA_API_KEY - see README)"
)

# Sample rate requested from Cartesia's API for raw PCM output. Cartesia
# supports several rates; 44100 is used consistently so every clip and voice
# shares one rate regardless of which model/voice generated it.
SAMPLE_RATE = 44100


def add_cli_arguments(parser) -> None:
    parser.add_argument(
        "--cartesia-model",
        type=str,
        default="sonic-2",
        help=(
            "Cartesia model id for --model cartesia (default: 'sonic-2', a pinned "
            "stable model rather than a moving 'latest' alias, since batches should "
            "stay reproducible). See play.cartesia.ai for other available models."
        ),
    )


def validate_args(args, parser) -> None:
    if args.all_fr or args.all_eng:
        parser.error(
            "--all-fr/--all-eng are --model kyutai only; use --all-voices for "
            "--model cartesia (it loops over your Cartesia account's voice "
            "library instead of a language-split repo)."
        )
    if not args.all_voices and not args.voice:
        parser.error(
            "--model cartesia requires --voice <voice_id> (copy one from "
            "play.cartesia.ai), or --all-voices to loop over your account's "
            "whole voice library."
        )
    if not os.environ.get("CARTESIA_API_KEY"):
        parser.error(
            "CARTESIA_API_KEY is not set. Get a key from play.cartesia.ai and "
            "export it as an environment variable before running with "
            "--model cartesia."
        )


def load_client():
    """Lazily construct the Cartesia API client.

    Imported here rather than at module scope so that installing the
    `cartesia` package (a separate, cloud-API-only dependency - see README)
    is only required when --model cartesia is actually used. Assumes
    CARTESIA_API_KEY has already been validated present (see validate_args).
    """
    from cartesia import Cartesia

    return Cartesia(api_key=os.environ["CARTESIA_API_KEY"])


def synthesize_clip(client, voice_id: str, model_id: str, language: str, text: str) -> np.ndarray:
    """Run one line of text through Cartesia's cloud API and return float32 PCM."""
    chunks = []
    stream = client.tts.generate_sse(
        model_id=model_id,
        transcript=text,
        voice=voice_id,
        language=language,
        output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": SAMPLE_RATE,
        },
    )
    for event in stream:
        if event.type == "chunk" and event.audio:
            chunks.append(event.audio)
        elif event.type == "error":
            raise RuntimeError(f"Cartesia error: {event.title}: {event.message}")
    return np.frombuffer(b"".join(chunks), dtype=np.float32)


def list_all_voices(client) -> list[tuple[str, str]]:
    """Every (voice_id, name) pair in this Cartesia account's voice library."""
    return sorted(
        ((voice.id, voice.name) for voice in client.voices.list()),
        key=lambda pair: pair[1],
    )


def run_single_voice(args, groups) -> None:
    client = load_client()
    synthesize_fn = functools.partial(
        synthesize_clip, client, args.voice, args.cartesia_model, args.language
    )
    audio.generate_groups_for_voice(
        synthesize_fn, SAMPLE_RATE, groups, args.output_dir, args.gap_ms
    )


def run_all_voices(args, groups) -> None:
    client = load_client()
    voices = list_all_voices(client)
    if args.voice_limit:
        voices = voices[: args.voice_limit]

    clips_per_voice = sum(len(lines) for _, lines in groups)
    total_clips = clips_per_voice * len(voices)
    message = (
        f"{len(voices)} voices x {clips_per_voice} clips each = {total_clips} clips.\n"
        f"Each clip is a call to Cartesia's cloud API ('{args.cartesia_model}' model), "
        "consuming your account's credits/quota - check play.cartesia.ai for current "
        "pricing before running this at scale."
    )
    if not audio.confirm_or_abort(message, args.yes):
        return

    audio.write_voices_manifest(
        args.output_dir, [f"{voice_id} ({name})" for voice_id, name in voices]
    )

    for index, (voice_id, name) in enumerate(tqdm.tqdm(voices, desc="voices"), start=1):
        try:
            synthesize_fn = functools.partial(
                synthesize_clip, client, voice_id, args.cartesia_model, args.language
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
            # A single bad/rate-limited voice shouldn't abort a run spanning
            # the whole account's voice library.
            print(f"Skipping voice {name!r} ({voice_id}) after error: {e!r}")


def interactive_args(common: dict) -> list[str]:
    argv: list[str] = []
    print("\n-- Cartesia options --")
    print("(Requires CARTESIA_API_KEY to already be set in this environment.)")
    all_voices = (
        input("Generate for every voice in your account's library? [y/N]: ").strip().lower()
        == "y"
    )
    if all_voices:
        argv.append("--all-voices")
    else:
        voice_id = input("Voice ID (copy one from play.cartesia.ai): ").strip()
        while not voice_id:
            voice_id = input("Voice ID (copy one from play.cartesia.ai): ").strip()
        argv += ["--voice", voice_id]

    model = input("Cartesia model id [sonic-2]: ").strip() or "sonic-2"
    argv += ["--cartesia-model", model]
    return argv
