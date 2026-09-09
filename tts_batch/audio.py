from pathlib import Path
from typing import Callable

import numpy as np
import sphn
import tqdm


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
    this same resumable/skip logic is shared by every backend (see each
    backend's run_single_voice/run_all_voices).

    Skipping existing output files (rather than always overwriting) is what
    makes an interrupted --all-voices run resumable: rerunning the same
    command only fills in what's missing instead of starting over.

    `filename_suffix` disambiguates multiple voices writing into the same
    flat `output_dir` (see each backend's run_all_voices); it's empty for
    the single-voice case, where <group>.wav alone is unambiguous.
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


def write_voices_manifest(output_dir: Path, voice_files: list[str]) -> None:
    """Index -> source voice path, since flat multi-voice filenames (e.g.
    'intro47.wav') can't carry that information themselves."""
    manifest_path = output_dir / "voices_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for index, voice_file in enumerate(voice_files, start=1):
            manifest.write(f"{index}\t{voice_file}\n")
    print(f"Wrote voice index -> source mapping to {manifest_path}")


def confirm_or_abort(message: str, yes: bool) -> bool:
    """Print `message`, then ask to continue unless `yes`.

    Shared by every backend's run_all_voices: a big run (hundreds of voices,
    hours of compute, or paid API calls) prints an estimate/cost note and
    waits for confirmation, unless --yes was passed (needed for
    non-interactive runs, e.g. under nohup). Returns True if the run should
    proceed.
    """
    print(message)
    if yes:
        return True
    if input("Continue? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return False
    return True
