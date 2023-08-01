import logging
import sys
import time
from pathlib import Path
from typing import Optional, Self

import numpy as np
import taichi as ti

from .camera import Camera


class Window(ti.ui.Window):
    frame_count: int = 0
    frame_duration: float = 0.0
    frame_interval: int = 1  # write image every `frame_interval` frames
    start_time: float = 0.0
    last_frame_time: float = 0.0
    show_window: bool = True
    video_manager: Optional[ti.tools.VideoManager] = None

    def __init__(
        self,
        name: str = "Hello, Taichi!",
        res: tuple[int, int] = (800, 600),
        vsync: bool = False,
        show_window: bool = True,
        fps_limit: int = 1000,
        pos: tuple[int, int] = (100, 100),
        *,
        output_dir: Path = Path.cwd(),
        frame_interval: int = 1,
        framerate: int = 30,
        automatic_build: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            res=res,
            vsync=vsync,
            show_window=show_window,
            fps_limit=fps_limit,
            pos=pos,
        )
        self.frame_interval = frame_interval
        self.show_window = show_window
        if self.frame_interval > 0:
            self.video_manager = ti.tools.VideoManager(
                output_dir=output_dir,
                framerate=framerate,
                automatic_build=automatic_build,
            )

    def __enter__(self) -> Self:
        self.frame_count = -1
        self.start_time = time.perf_counter()
        self.last_frame_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.running = False
        fps: float = self.frame_count / (self.last_frame_time - self.start_time)
        logging.info(f"FPS: {fps:.2f}")
        if self.frame_interval > 0:
            assert self.video_manager is not None
            self.video_manager.make_video(mp4=True, gif=True)

    def next_frame(
        self,
        *,
        max_frames: int = sys.maxsize,
        track_user_input: Optional[Camera] = None,
    ) -> bool:
        if max_frames and (self.frame_count >= max_frames):
            return False

        if self.frame_interval > 0 and self.frame_count >= 0:
            assert self.video_manager is not None
            image_buffer: np.ndarray = self.get_image_buffer_as_numpy()
            self.video_manager.write_frame(image_buffer)

        if self.show_window:
            if track_user_input:
                track_user_input.track_user_inputs(window=self)
            self.show()

        self.frame_count += 1
        self.frame_duration = time.perf_counter() - self.last_frame_time
        self.last_frame_time = time.perf_counter()

        return self.running
