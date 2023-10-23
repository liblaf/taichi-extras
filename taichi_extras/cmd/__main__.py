from typer import Typer

from taichi_extras.common.typer import add_command

from .mesh.__main__ import app as cmd_mesh

app: Typer = Typer(name="te")
add_command(app=app, command=cmd_mesh, name="mesh")

if __name__ == "__main__":
    from taichi_extras.common.typer import run

    run(app)
