import typing
from typing import Any


def cast(typ: Any, val: Any) -> Any:
    typ = typing.get_origin(typ) or typ
    try:
        assert isinstance(val, typ)
    except TypeError:
        pass
    return val
