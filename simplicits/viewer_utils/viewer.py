"""
Modified from Shape of Motion (https://github.com/vye16/shape-of-motion).
"""

from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, Viewer
from viser import Icon, ViserServer

from .playback_panel import add_gui_playback_group
from .render_panel import populate_render_tab


class DynamicViewer(Viewer):
    def __init__(
        self,
        server: ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        num_frames: int,
        work_dir: str,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        self.num_frames = max(1, num_frames)
        self.work_dir = Path(work_dir)
        super().__init__(server, render_fn, mode)

    def _define_guis(self):
        super()._define_guis()
        server = self.server

        with server.gui.add_folder(
            "Time", order=self._rendering_folder.order + 0.1
        ) as self._time_folder:
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,
                initial_fps=15.0,
            )
            self._playback_guis[0].on_update(self.rerender)
            self._canonical_checkbox = server.gui.add_checkbox("Canonical", False)
            self._canonical_checkbox.on_update(self.rerender)

            _cached_playback_disabled = []

            def _toggle_gui_playing(event):
                if event.target.value:
                    nonlocal _cached_playback_disabled
                    _cached_playback_disabled = [
                        gui.disabled for gui in self._playback_guis
                    ]
                    target_disabled = [True] * len(self._playback_guis)
                else:
                    target_disabled = _cached_playback_disabled
                for gui, disabled in zip(self._playback_guis, target_disabled):
                    gui.disabled = disabled

            self._canonical_checkbox.on_update(_toggle_gui_playing)

        self._point_checkbox = server.gui.add_checkbox(
            "Point Cloud", False, order=self._time_folder.order + 0.2
        )
        self._point_checkbox.on_update(self.rerender)
        self._render_track_checkbox = server.gui.add_checkbox(
            "Render tracks", False, order=self._time_folder.order + 0.2
        )
        self._render_track_checkbox.on_update(self.rerender)

        tabs = server.gui.add_tab_group(order=self._render_track_checkbox.order + 0.1)
        with tabs.add_tab("Render", Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                server, Path(self.work_dir) / "camera_paths", self._playback_guis[0]
            )
