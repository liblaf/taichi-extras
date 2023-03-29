import math
import time

import taichi as ti

from .utils import euler_to_vec, vec_to_euler


def track_user_inputs(
    self: ti.ui.Camera,
    window: ti.ui.Window,
    movement_speed: float = 60.0,
    yaw_speed: float = 120.0,
    pitch_speed: float = 120.0,
    hold_key=None,
):
    """Move the camera according to user inputs.
    Press `w`, `s`, `a`, `d`, `e`, `q` to move the camera
    `formard`, `back`, `left`, `right`, `head up`, `head down`, accordingly.

    Args:

        window (:class:`~taichi.ui.Window`): a windown instance.
        movement_speed (:mod:`~taichi.types.primitive_types`): camera movement speed.
        yaw_speed (:mod:`~taichi.types.primitive_types`): speed of changes in yaw angle.
        pitch_speed (:mod:`~taichi.types.primitive_types`): speed of changes in pitch angle.
        hold_key (:mod:`~taichi.ui`): User defined key for holding the camera movement.
    """
    front = (self.curr_lookat - self.curr_position).normalized()
    position_change = ti.Vector([0.0, 0.0, 0.0])
    left = self.curr_up.cross(front)
    up = self.curr_up

    if (not hasattr(self, "last_time")) or (self.last_time is None):
        self.last_time = time.perf_counter_ns()
    time_elapsed = (time.perf_counter_ns() - self.last_time) * 1e-9
    self.last_time = time.perf_counter_ns()

    movement_speed *= time_elapsed
    if window.is_pressed("w"):
        position_change += front * movement_speed
    if window.is_pressed("s"):
        position_change -= front * movement_speed
    if window.is_pressed("a"):
        position_change += left * movement_speed
    if window.is_pressed("d"):
        position_change -= left * movement_speed
    if window.is_pressed("e"):
        position_change += up * movement_speed
    if window.is_pressed("q"):
        position_change -= up * movement_speed
    self.position(*(self.curr_position + position_change))
    self.lookat(*(self.curr_lookat + position_change))

    curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
    if (hold_key is None) or window.is_pressed(hold_key):
        if (self.last_mouse_x is None) or (self.last_mouse_y is None):
            self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
        dx = curr_mouse_x - self.last_mouse_x
        dy = curr_mouse_y - self.last_mouse_y

        yaw, pitch = vec_to_euler(front)

        yaw_speed *= time_elapsed
        pitch_speed *= time_elapsed
        yaw -= dx * yaw_speed
        pitch += dy * pitch_speed

        pitch_limit = math.pi / 2 * 0.99
        if pitch > pitch_limit:
            pitch = pitch_limit
        elif pitch < -pitch_limit:
            pitch = -pitch_limit

        front = euler_to_vec(yaw, pitch)
        self.lookat(*(self.curr_position + front))

    self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
