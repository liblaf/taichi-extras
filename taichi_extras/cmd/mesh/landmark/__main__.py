from typer import Typer

from taichi_extras.common.typer import add_command

from .align import main as cmd_align
from .dense import main as cmd_dense
from .view import main as cmd_view

app: Typer = Typer(name="landmark")
add_command(app=app, command=cmd_align, name="align")
add_command(app=app, command=cmd_dense, name="dense")
add_command(app=app, command=cmd_view, name="view")


if __name__ == "__main__":
    from taichi_extras.common.typer import run

    run(app)
