from collections.abc import Iterable


def minify(lines: Iterable[str]) -> Iterable[str]:
    lines = map(lambda s: s.partition("#")[0], lines)
    lines = map(lambda s: s.strip(), lines)
    lines = filter(lambda s: s, lines)
    return lines
