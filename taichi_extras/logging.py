import logging
from logging import Formatter

import trimesh.util
from rich.logging import RichHandler


def init(level: int) -> None:
    handler: RichHandler = RichHandler(level=level)
    trimesh.util.attach_to_log(level=level, handler=handler)
    logging.basicConfig(
        format="%(name)s: %(message)s",
        datefmt="[%x]",
        level=level,
        handlers=[handler],
    )
    logging.getLogger("taichi_extras")
    handler.setFormatter(Formatter("%(name)s: %(message)s"))
