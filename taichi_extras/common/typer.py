from typing import Callable, Optional

import typer
from typer import Typer

from . import logging


def run(cmd: Callable) -> None:
    logging.init()
    if isinstance(cmd, Typer):
        cmd()
    else:
        typer.run(function=cmd)


def add_command(app: Typer, command: Callable, name: Optional[str] = None) -> None:
    if isinstance(command, Typer):
        app.add_typer(command, name=name)
    else:
        app.command(name=name)(command)
