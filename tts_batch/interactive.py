"""Guided setup wizard, used when the script is run with no CLI arguments.

Collects answers via input(), builds the equivalent argv list, and hands it
back to cli.main() to parse with the very same argparse parser/validation
used by the direct-CLI path - so there's no separate validation logic to
keep in sync, and invalid answers surface the same parser.error() messages
a mistyped flag would.
"""

from pathlib import Path

from .backends import BACKENDS


def run_wizard(parser) -> list[str]:
    print(
        "No arguments given - starting the guided setup.\n"
        "(Run with --help instead for the full flag reference.)\n"
    )

    default_input = "input.example.txt" if Path("input.example.txt").exists() else None
    prompt = f"Input file [{default_input}]: " if default_input else "Input file: "
    input_path = input(prompt).strip() or default_input
    while not input_path:
        input_path = input("Input file: ").strip()
    argv = [input_path]

    print("\nBackends:")
    names = list(BACKENDS)
    for i, name in enumerate(names, start=1):
        print(f"  [{i}] {name} - {BACKENDS[name].DESCRIPTION}")
    while True:
        choice = input(f"Choose a backend [1-{len(names)}] (default 1): ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            model = names[int(choice) - 1]
            break
        print(f"Enter a number from 1 to {len(names)}.")
    argv += ["--model", model]

    argv += BACKENDS[model].interactive_args({})

    default_output_dir = parser.get_default("output_dir")
    output_dir = input(f"\nOutput directory [{default_output_dir}]: ").strip() or str(
        default_output_dir
    )
    argv += ["--output-dir", output_dir]

    default_gap_ms = parser.get_default("gap_ms")
    gap_ms = input(f"Gap between clips in ms [{default_gap_ms}]: ").strip() or str(default_gap_ms)
    argv += ["--gap-ms", gap_ms]

    print("\nEquivalent command:")
    print("  python generate_and_concat.py " + " ".join(argv))
    print()

    return argv
