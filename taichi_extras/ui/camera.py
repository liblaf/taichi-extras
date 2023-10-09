import time
from typing import cast

import numpy as np
import taichi as ti
from taichi import Vector
from taichi.lang import matrix_ops

from .utils import clamp, euler_to_vec, vec_to_euler


class Camera(ti.ui.Camera):
    last_mouse_x: float = np.nan
    last_mouse_y: float = np.nan
    last_time: float = np.nan

    def track_user_inputs(
        self,
        window: ti.ui.Window,
        movement_speed: float = 1.0,
        yaw_speed: float = 1.0,
        pitch_speed: float = 1.0,
        hold_key: str = ti.ui.LMB,
    ) -> None:
        """
        Move the camera according to user inputs.
        Press `w`, `s`, `a`, `d`, `space`, `shift` to move the camera `forward`, `back`, `left`, `right`, `up`, `down`, accordingly.

        Args:
            window (ti.ui.Window): A window instance.
            movement_speed (float, optional): Camera movement speed. Defaults to 1.0.
            yaw_speed (float, optional): Speed of changes in yaw angle. Defaults to 1.0.
            pitch_speed (float, optional): Speed of changes in pitch angle. Defaults to 1.0.
            hold_key (str, optional): User defined key for holding the camera movement. Defaults to ti.ui.LMB.
        """
        sight: Vector = matrix_ops.normalized(self.curr_lookat - self.curr_position)
        left: Vector = matrix_ops.normalized(self.curr_up.cross(sight))
        front: Vector = matrix_ops.normalized(left.cross(self.curr_up))
        up: Vector = self.curr_up

        self.last_time = self.last_time or time.perf_counter()
        time_elapsed: float = time.perf_counter() - self.last_time
        self.last_time = time.perf_counter()

        position_change: Vector = Vector([0.0, 0.0, 0.0])
        distance: float = movement_speed * time_elapsed
        if window.is_pressed("w"):
            position_change += distance * front
        if window.is_pressed("s"):
            position_change -= distance * front
        if window.is_pressed("a"):
            position_change += distance * left
        if window.is_pressed("d"):
            position_change -= distance * left
        if window.is_pressed(ti.ui.SPACE):
            position_change += distance * up
        if window.is_pressed(ti.ui.SHIFT):
            position_change -= distance * up
        self.position(*cast(Vector, self.curr_position + position_change))
        self.lookat(*cast(Vector, self.curr_lookat + position_change))

        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
        if (hold_key is None) or window.is_pressed(hold_key):
            if (self.last_mouse_x is None) or (self.last_mouse_y is None):
                self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
            dx: float = curr_mouse_x - self.last_mouse_x
            dy: float = curr_mouse_y - self.last_mouse_y
            yaw, pitch = vec_to_euler(sight)
            yaw += dx * yaw_speed
            pitch -= dy * pitch_speed
            pitch_limit: float = np.deg2rad(89.0)
            pitch: float = clamp(pitch, -pitch_limit, pitch_limit)
            sight = euler_to_vec(yaw, pitch)
            self.lookat(*cast(Vector, self.curr_position + sight))

        self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
