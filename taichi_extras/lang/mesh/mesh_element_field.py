from typing import Any

from taichi import Layout
from taichi.lang.mesh import MeshElementField


def place(
    field: MeshElementField,
    members: dict[str, Any],
    reorder: bool = False,
    needs_grad: bool = False,
    layout: Any = Layout.SOA,
) -> None:
    for key in list(members.keys()):
        if key in field.keys:
            del members[key]
    field.place(members, reorder=reorder, needs_grad=needs_grad, layout=layout)
