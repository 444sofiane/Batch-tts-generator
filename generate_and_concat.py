import argparse
import re
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
    tts_model: TTSModel,
    condition_attributes,
    groups: list[tuple[str, list[str]]],
    output_dir: Path,
    gap_ms: float,
    filename_suffix: str = "",
) -> None:
    """Generate+concatenate every group for one voice, skipping already-done groups.

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
            synthesize_clip(tts_model, condition_attributes, text)
            for text in tqdm.tqdm(lines, desc=group_name, leave=False)
        ]
        audio = concatenate_clips(clips, tts_model.mimi.sample_rate, gap_ms)
        sphn.write_wav(str(out_path), audio, tts_model.mimi.sample_rate)
        print(f"Wrote {out_path} ({len(audio) / tts_model.mimi.sample_rate:.1f}s)")


def run_single_voice(args: argparse.Namespace, tts_model: TTSModel, groups) -> None:
    voice = args.voice or DEFAULT_VOICE_BY_LANGUAGE[args.language]
    voice_path = voice if voice.endswith(".safetensors") else tts_model.get_voice_path(voice)
    condition_attributes = tts_model.make_condition_attributes([voice_path], cfg_coef=2.0)
    generate_groups_for_voice(tts_model, condition_attributes, groups, args.output_dir, args.gap_ms)


def write_voices_manifest(output_dir: Path, voice_files: list[str]) -> None:
    """Index -> source voice path, since flat multi-voice filenames (e.g.
    'intro47.wav') can't carry that information themselves."""
    manifest_path = output_dir / "voices_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for index, voice_file in enumerate(voice_files, start=1):
            manifest.write(f"{index}\t{voice_file}\n")
    print(f"Wrote voice index -> source mapping to {manifest_path}")


def run_all_voices(args: argparse.Namespace, tts_model: TTSModel, groups) -> None:
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
            generate_groups_for_voice(
                tts_model,
                condition_attributes,
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


def run(args: argparse.Namespace) -> None:
    groups = parse_input(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=32, temp=0.6, device=args.device
    )

    if args.all_voices or args.all_fr or args.all_eng:
        run_all_voices(args, tts_model, groups)
    else:
        run_single_voice(args, tts_model, groups)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS clips with Kyutai's PyTorch TTS and concatenate them per group."
    )
    parser.add_argument("input", type=Path, help="Path to the input .txt file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Where to write .wav files"
    )
    parser.add_argument(
        "--hf-repo", type=str, default=DEFAULT_DSM_TTS_REPO, help="HF repo for the TTS model"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=sorted(DEFAULT_VOICE_BY_LANGUAGE),
        default="en",
        help="Picks a default voice for this language (ignored if --voice is set)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help=(
            "Voice to use (overrides --language): a path under the voice repo root "
            "WITHOUT the trailing .<hash>@<epoch>.safetensors suffix (e.g. "
            "'cml-tts/fr/10087_11650_000028-0002.wav') to fetch from Hugging Face, "
            "or a path to a .safetensors file already on disk."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device to run on (default: cpu)",
    )
    parser.add_argument(
        "--gap-ms", type=float, default=300.0, help="Pause between clips within a group"
    )
    parser.add_argument(
        "--all-voices",
        action="store_true",
        help=(
            "Generate every group for every voice in the voice repo (901+ voices), "
            "instead of just one. Ignores --voice/--language. Writes to "
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
            "(the French voices) instead of the whole repo."
        ),
    )
    parser.add_argument(
        "--all-eng",
        action="store_true",
        help=(
            f"Like --all-voices, but restricted to voices under '{EN_VOICES_PREFIX}' "
            "(the English Expresso voices) instead of the whole repo."
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
    run(args)


if __name__ == "__main__":
    main()
