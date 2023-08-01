import math
import time

import numpy as np
import taichi as ti

from .utils import euler_to_vec, vec_to_euler


class Camera(ti.ui.Camera):
    last_mouse_x: float = np.nan
    last_mouse_y: float = np.nan
    last_time: float = np.nan

    def track_user_inputs(
        self: ti.ui.Camera,
        window: ti.ui.Window,
        movement_speed: float = 1.0,
        yaw_speed: float = 1.0,
        pitch_speed: float = 1.0,
        hold_key=ti.ui.LMB,
    ):
        """
        Move the camera according to user inputs.
        Press `w`, `s`, `a`, `d`, `Space`, `Shift` to move the camera `forward`, `back`, `left`, `right`, `up`, `down`, accordingly.

        Parameters:
            window: a window instance.
            movement_speed: camera movement speed.
            yaw_speed: speed of changes in yaw angle.
            pitch_speed: speed of changes in pitch angle.
            hold_key: User defined key for holding the camera movement.
        """
        sight: ti.Vector = (self.curr_lookat - self.curr_position).normalized()  # type: ignore
        left: ti.Vector = self.curr_up.cross(sight).normalized()  # type: ignore
        front: ti.Vector = left.cross(self.curr_up).normalized()  # type: ignore
        up: ti.Vector = self.curr_up

        self.last_time = self.last_time or time.perf_counter_ns()
        time_elapsed: float = (time.perf_counter_ns() - self.last_time) * 1e-9
        self.last_time = time.perf_counter_ns()

        position_change: ti.Vector = ti.Vector([0.0, 0.0, 0.0])
        movement_speed *= time_elapsed
        if window.is_pressed("w"):
            position_change += front * movement_speed
        if window.is_pressed("s"):
            position_change -= front * movement_speed
        if window.is_pressed("a"):
            position_change += left * movement_speed
        if window.is_pressed("d"):
            position_change -= left * movement_speed
        if window.is_pressed(ti.ui.SPACE):
            position_change += up * movement_speed
        if window.is_pressed(ti.ui.SHIFT):
            position_change -= up * movement_speed
        self.position(*(self.curr_position + position_change))  # type: ignore
        self.lookat(*(self.curr_lookat + position_change))  # type: ignore

        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
        if (hold_key is None) or window.is_pressed(hold_key):
            if (self.last_mouse_x is None) or (self.last_mouse_y is None):
                self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
            dx: float = curr_mouse_x - self.last_mouse_x
            dy: float = curr_mouse_y - self.last_mouse_y
            yaw: float
            pitch: float
            yaw, pitch = vec_to_euler(sight)
            yaw += dx * yaw_speed
            pitch -= dy * pitch_speed
            pitch_limit: float = np.deg2rad(89.0)
            if pitch > pitch_limit:
                pitch = pitch_limit
            elif pitch < -pitch_limit:
                pitch = -pitch_limit

            sight = euler_to_vec(yaw, pitch)
            self.lookat(*(self.curr_position + sight))  # type: ignore

        self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
