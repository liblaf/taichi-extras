import os
import time
import typing
from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti
import typing_extensions


class Window:
    frame_count: int = 0
    frame_duration: float = 0.0
    frame_interval: Optional[int] = 1  # write image every `frame_interval` frames
    previous_frame_time: float = 0.0
    video_manager: Optional[ti.tools.VideoManager] = None
    window: ti.ui.Window

    def __init__(self, window: ti.ui.Window, frame_interval: Optional[int] = 1) -> None:
        assert (frame_interval is None) or (frame_interval > 0)
        self.frame_interval = frame_interval
        if self.frame_interval is not None:
            self.video_manager = ti.tools.VideoManager(output_dir=Path.cwd())
        self.window = window

    def __enter__(self) -> typing_extensions.Self:
        self.previous_frame_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.window.running = False
        if self.frame_interval is not None:
            assert self.video_manager is not None
            self.video_manager.make_video(mp4=True, gif=True)

    @property
    def running(self) -> bool:
        return self.window.running

    def next_frame(self, max_frames: typing.Optional[int] = None) -> bool:
        if max_frames and (self.frame_count >= max_frames):
            return False
        self.frame_count += 1
        self.frame_duration = time.perf_counter() - self.previous_frame_time
        self.previous_frame_time = time.perf_counter()

        if self.frame_interval:
            assert self.video_manager is not None
            image_buffer: np.ndarray = self.window.get_image_buffer_as_numpy()
            self.video_manager.write_frame(image_buffer)

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
