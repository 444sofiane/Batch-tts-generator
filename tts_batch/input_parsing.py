import re
from pathlib import Path

GROUP_HEADER_RE = re.compile(r"^#\s*(.+)$")


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
