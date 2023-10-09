import logging
import sys
import time
from pathlib import Path
from typing import Optional, Self

import numpy as np
import taichi as ti
from taichi.tools import VideoManager

from .camera import Camera


class Window(ti.ui.Window):
    frame_count: int = 0
    frame_duration: float = 0.0
    last_frame_time: float = 0.0
    show_window: bool = True
    start_time: float = 0.0
    video_manager: Optional[VideoManager] = None

    def __init__(
        self,
        name: str = "Hello, Taichi!",
        res: tuple[int, int] = (1920, 1440),
        vsync: bool = False,
        show_window: bool = True,
        fps_limit: int = 1000,
        pos: tuple[int, int] = (100, 100),
        *,
        automatic_build: bool = False,
        framerate: int = 30,
        output_dir: Optional[Path] = Path.cwd() / "video",
    ) -> None:
        super().__init__(
            name=name,
            res=res,
            vsync=vsync,
            show_window=show_window,
            fps_limit=fps_limit,
            pos=pos,
        )
        self.show_window = show_window
        if output_dir:
            self.video_manager = VideoManager(
                output_dir=output_dir,
                framerate=framerate,
                automatic_build=automatic_build,
            )

    def __enter__(self) -> Self:
        self.frame_count = -1
        self.last_frame_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.running = False
        fps: float = self.frame_count / (self.last_frame_time - self.start_time)
        logging.info(f"FPS Average: {fps:.2f}")
        if self.video_manager:
            self.video_manager.make_video(mp4=True, gif=False)

    def next_frame(
        self,
        *,
        max_frames: int = sys.maxsize,
        track_user_input: Optional[Camera] = None,
    ) -> bool:
        if self.show_window:
            self.show()
            if track_user_input:
                track_user_input.track_user_inputs(window=self)
        if self.video_manager and self.frame_count >= 0:
            image_buffer: np.ndarray = self.get_image_buffer_as_numpy()
            self.video_manager.write_frame(image_buffer)

        # new frame

        self.frame_count += 1
        if self.frame_count == 0:
            self.start_time = time.perf_counter()
        self.frame_duration = time.perf_counter() - self.last_frame_time
        self.last_frame_time = time.perf_counter()
        if max_frames and (self.frame_count >= max_frames):
            return False

        return self.running
