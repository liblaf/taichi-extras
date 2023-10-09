from collections.abc import Callable

from typer import Typer


def run(main: Callable, pretty_exceptions_show_locals: bool = False) -> None:
    app: Typer = Typer(pretty_exceptions_show_locals=pretty_exceptions_show_locals)
    app.command()(main)
    app()
