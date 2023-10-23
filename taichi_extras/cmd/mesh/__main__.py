from typer import Typer

from taichi_extras.common.typer import add_command

from .info import main as cmd_info
from .landmark.__main__ import app as cmd_landmark
from .simplify import main as cmd_simplify

app: Typer = Typer(name="mesh")
add_command(app=app, command=cmd_info, name="info")
add_command(app=app, command=cmd_landmark, name="landmark")
add_command(app=app, command=cmd_simplify, name="simplify")

if __name__ == "__main__":
    from taichi_extras.common.typer import run

    run(app)
