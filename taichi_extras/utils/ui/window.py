import os
import time
from pathlib import Path

import taichi as ti
import typing_extensions
import typing


class Window:
    frame_count: int = 0
    frame_duration: float = 0.0
    previous_frame_time: float = 0.0

    def __init__(self, window: ti.ui.Window) -> None:
        self.window = window

    def __enter__(self) -> typing_extensions.Self:
        self.previous_frame_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.window.running = False

    @property
    def running(self) -> bool:
        return self.window.running

    def next_frame(self, max_frames: typing.Optional[int] = None) -> bool:
        if max_frames and (self.frame_count >= max_frames):
            return False
        self.frame_count += 1
        self.frame_duration = time.perf_counter() - self.previous_frame_time
        self.previous_frame_time = time.perf_counter()
        return self.running

    def save_image(
        self, interval: int = 1, prefix: Path = Path.cwd(), stem: str = "frame"
    ) -> bool:
        if self.frame_count % interval == 0:
            idx: int = self.frame_count // interval
            if not prefix.exists():
                os.makedirs(name=prefix, exist_ok=True)
            filepath: Path = prefix / f"{stem}-{idx:08d}.png"
            self.window.save_image(str(filepath))
            return True
        return False
