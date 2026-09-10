"""Registry of every supported TTS backend, keyed by --model name.

Dict order here is display order: it decides both the order backend-
specific flags appear in --help (see cli.build_parser) and the order
choices are numbered in the interactive wizard's backend menu (see
interactive.run_wizard).
"""

from . import breeze, cartesia, kyutai, piper, tortoise, xtts

BACKENDS = {
    kyutai.NAME: kyutai,
    tortoise.NAME: tortoise,
    breeze.NAME: breeze,
    cartesia.NAME: cartesia,
    piper.NAME: piper,
    xtts.NAME: xtts,
}
