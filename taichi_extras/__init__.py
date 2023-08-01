import logging

from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s", datefmt="[%x]", level=logging.INFO, handlers=[RichHandler()]
)
