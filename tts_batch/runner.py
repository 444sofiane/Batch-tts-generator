"""Dispatches a fully-parsed/validated args.Namespace to the selected backend."""

from . import input_parsing
from .backends import BACKENDS


def run(args) -> None:
    groups = input_parsing.parse_input(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = BACKENDS[args.model]
    if args.all_voices or args.all_fr or args.all_eng:
        backend.run_all_voices(args, groups)
    else:
        backend.run_single_voice(args, groups)
