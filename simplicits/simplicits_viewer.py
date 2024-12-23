from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import gsplat as gs
import kaolin as kal
import numpy as np
import torch
import trimesh
import tyro
import viser
import viser.transforms
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfview import CameraState
from plyfile import PlyData

from .trainer.model import MLP
from .trainer.train import load_model
from .utils import (
    build_scaling_rotation,
    compute_adaptive_opacity_threshold,
    kaolin_to_glb_binary,
    remove_outliers,
    seed_all,
)
from .viewer_utils.utils import get_server
from .viewer_utils.viewer import DynamicViewer


@dataclass
class RunViewer:
    """Load a trained Simplicits model and start the viewer."""

    # Type of the scene we are working with, either: "mesh", "nerf", or "gs"
    base_type: str = "mesh"
    # Path to the original geometry. (1) If base_type is "mesh", this should
    # be a .obj file. (2) If base_type is "nerf", this should be a .yaml
    # config file. (3) If base_type is "gs", this should be a .ply file.
    original_geometry: Path = Path("")
    # Geometry Parameters - Density threshold for the NeRF model or the opacity
    # threshold for the GaussianSplat. If not provided, we will compute an
    # adaptive threshold.
    density_threshold: Optional[float] = None
    # Path to the trained model
    model_path: Path = Path("model.safetensors")
    # Training parameters - device to train on
    device: Optional[str | torch.device] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Training parameters - number of handles for the model for simulation,
    # default from paper supplementary for popular splats
    handles: int = 40
    # Training parameters - number of layers in the MLP model
    layers: int = 9
    # Training parameters - number of points to sample from the geometry
    num_samples: int = int(1e6)
    # Number of simulation steps
    num_steps: int = 100
    # Physical material properties - Young's modulus i.e. stiffness
    soft_youngs_modulus: float = 1e5
    # Physical material properties - Poisson's ratio i.e. ratio of lateral
    # strain to longitudinal strain
    poisson_ratio: float = 0.45
    # Physical material properties - Density
    rho: float = 100
    # Physical material properties - Approximate volume of the object
    approx_volume: float = 1
    # Scene parameters - Floor height
    floor_height: float = -0.8
    # Scene parameters - Penalty for the floor
    floor_penalty: float = 1000
    # Scene parameters - Gravity
    gravity: List[float] = field(default_factory=lambda: [0, 9.8, 0])

    def main(self) -> None:
        """Main function."""
        seed_all(3407)

        self.setup_simulation()
        self.simulate_and_cache_frames()

        self.server = get_server(port=7007)
        if self.base_type == "mesh":
            self.viewer = DynamicViewer(
                self.server,
                self.render_fn_mesh,
                num_frames=max(1, len(self.cached_frames)),
                work_dir="./",
                mode="rendering",
            )
        elif self.base_type == "gs":
            self.viewer = DynamicViewer(
                self.server,
                self.render_fn_gs,
                num_frames=max(1, len(self.cached_frames)),
                work_dir="./",
                mode="rendering",
            )
        elif self.base_type == "nerf":
            self.viewer = DynamicViewer(
                self.server,
                self.render_fn_nerf,
                num_frames=max(1, len(self.cached_frames)),
                work_dir="./",
                mode="rendering",
            )
        self.setup_scene_controls(self.server)

    def setup_scene_controls(self, server: viser.ViserServer) -> None:
        scene_folder = server.gui.add_folder(
            "Scene Parameters", order=self.viewer._time_folder.order + 0.1
        )
        with scene_folder:
            self.floor_enabled = server.gui.add_checkbox(
                "Enable Floor", initial_value=True
            )
            self.floor_axis = server.gui.add_dropdown(
                "Floor Axis", options=["X", "Y", "Z"], initial_value="Y"
            )
            self.floor_height_slider = server.gui.add_slider(
                "Floor Height",
                min=-2.0,
                max=2.0,
                step=0.1,
                initial_value=self.floor_height,
            )
            self.floor_penalty_slider = server.gui.add_slider(
                "Floor Penalty",
                min=0,
                max=5000,
                step=100,
                initial_value=self.floor_penalty,
            )

            self.gravity_enabled = server.gui.add_checkbox(
                "Enable Gravity", initial_value=True
            )
            self.gravity_strength = server.gui.add_slider(
                "Gravity Strength",
                min=0,
                max=20,
                step=0.1,
                initial_value=9.8,
            )
            self.gravity_direction = server.gui.add_dropdown(
                "Gravity Direction",
                options=["Down (Y)", "Down (Z)", "Custom"],
                initial_value="Down (Y)",
            )

            self.gravity_x = server.gui.add_number(
                "Gravity X",
                min=-20,
                max=20,
                step=0.1,
                initial_value=0,
                visible=False,
            )
            self.gravity_y = server.gui.add_number(
                "Gravity Y",
                min=-20,
                max=20,
                step=0.1,
                initial_value=9.8,
                visible=False,
            )
            self.gravity_z = server.gui.add_number(
                "Gravity Z",
                min=-20,
                max=20,
                step=0.1,
                initial_value=0,
                visible=False,
            )

            self.reset_button = server.gui.add_button(
                "Reset Simulation",
                icon="refresh",
                color="red",
            )

            self.setup_event_handlers()

    def setup_event_handlers(self) -> None:
        @self.floor_enabled.on_update
        def _(_) -> None:
            self.floor_height_slider.disabled = not self.floor_enabled.value
            self.floor_penalty_slider.disabled = not self.floor_enabled.value
            self.floor_axis.disabled = not self.floor_enabled.value
            self.update_scene_parameters()

        @self.floor_height_slider.on_update
        def _(_) -> None:
            self.floor_height = self.floor_height_slider.value
            self.update_scene_parameters()

        @self.floor_axis.on_update
        def _(_) -> None:
            if self.floor_axis.value == "X":
                self.floor_axis_value = 0
            elif self.floor_axis.value == "Y":
                self.floor_axis_value = 1
            else:
                self.floor_axis_value = 2
            self.update_scene_parameters()

        @self.floor_penalty_slider.on_update
        def _(_) -> None:
            self.floor_penalty = self.floor_penalty_slider.value
            self.update_scene_parameters()

        @self.gravity_enabled.on_update
        def _(_) -> None:
            self.gravity_strength.disabled = not self.gravity_enabled.value
            self.gravity_direction.disabled = not self.gravity_enabled.value
            update_gravity_visibility()
            self.update_scene_parameters()

        @self.gravity_strength.on_update
        def _(_) -> None:
            self.update_gravity_vector()
            self.update_scene_parameters()

        @self.gravity_direction.on_update
        def _(_) -> None:
            update_gravity_visibility()
            self.update_gravity_vector()
            self.update_scene_parameters()

        @self.gravity_x.on_update
        @self.gravity_y.on_update
        @self.gravity_z.on_update
        def _(_) -> None:
            if self.gravity_direction.value == "Custom":
                self.gravity = [
                    self.gravity_x.value,
                    self.gravity_y.value,
                    self.gravity_z.value,
                ]
                self.update_scene_parameters()

        @self.reset_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            loading_notif = client.add_notification(
                title="Resetting Simulation",
                body="Simulation is being reset",
                loading=True,
                with_close_button=False,
                auto_close=False,
            )
            self.reset_simulation()
            self.simulate_and_cache_frames()
            loading_notif.title = "Simulation Reset"
            loading_notif.body = "Simulation has been reset"
            loading_notif.loading = False
            loading_notif.with_close_button = True
            loading_notif.auto_close = 5000
            loading_notif.color = "green"

        def update_gravity_visibility() -> None:
            is_custom = self.gravity_direction.value == "Custom"
            self.gravity_x.visible = is_custom
            self.gravity_y.visible = is_custom
            self.gravity_z.visible = is_custom

    def update_gravity_vector(self) -> None:
        if not self.gravity_enabled.value:
            self.gravity = [0, 0, 0]
        else:
            strength = self.gravity_strength.value
            if self.gravity_direction.value == "Down (Y)":
                self.gravity = [0, strength, 0]
            elif self.gravity_direction.value == "Down (Z)":
                self.gravity = [0, 0, strength]
            else:
                self.gravity = [
                    self.gravity_x.value,
                    self.gravity_y.value,
                    self.gravity_z.value,
                ]

    def update_scene_parameters(self) -> None:
        if self.floor_enabled.value:
            self.scene.set_scene_floor(
                floor_height=self.floor_height,
                floor_axis=self.floor_axis_value,
                floor_penalty=self.floor_penalty,
            )
        else:
            self.scene.set_scene_floor(
                floor_height=self.floor_height,
                floor_axis=1,
                floor_penalty=0,
            )

        self.scene.set_scene_gravity(
            acc_gravity=torch.tensor(self.gravity, device=self.device)
        )

    def setup_simulation(self) -> None:
        """Setup the initial simulation state."""
        self.cached_frames = []
        self.current_frame = 0

        if self.base_type == "mesh":
            self.mesh = kal.io.import_mesh(
                str(self.original_geometry), triangulate=True
            ).to(self.device)
            self.mesh.vertices = kal.ops.pointcloud.center_points(
                self.mesh.vertices.unsqueeze(0), normalize=True
            ).squeeze(0)
            self.orig_vertices = self.mesh.vertices.clone()

            uniform_pts = (
                torch.rand(self.num_samples, 3, device=self.device)
                * (
                    self.orig_vertices.max(dim=0).values
                    - self.orig_vertices.min(dim=0).values
                )
                + self.orig_vertices.min(dim=0).values
            )

            boolean_signs = kal.ops.mesh.check_sign(
                self.mesh.vertices.unsqueeze(0),
                self.mesh.faces,
                uniform_pts.unsqueeze(0),
                hash_resolution=512,
            )

            pts = uniform_pts[boolean_signs.squeeze()]

            # i originally started by using trimesh similar to how we train the model
            # however due to certain redundant reads and writes later during rendering
            # this version gets really slow, so i used the kaolin version. particularly
            # because finally when we render an image we anyways need to give viser a
            # tensor.

            # self.mesh = trimesh.load_mesh(
            #     str(self.original_geometry), force="mesh", process=False
            # )
            # unique_vertices, _ = np.unique(self.mesh.vertices, axis=0, return_inverse=True)
            # vertices = torch.tensor(unique_vertices, dtype=torch.float32)
            # vmin = vertices.min(dim=0, keepdim=True)[0]
            # vmax = vertices.max(dim=0, keepdim=True)[0]
            # vmid = (vmin + vmax) / 2
            # centered_vertices = vertices - vmid
            # den = (vmax - vmin).max().clip(min=1e-6)
            # centered_vertices = centered_vertices / den
            # bounds = self.mesh.bounds
            # uniform_samples = np.random.uniform(
            #     bounds[0], bounds[1], size=(self.num_samples, 3)
            # )
            # inside_mask = self.mesh.contains(uniform_samples)
            # pts = torch.tensor(
            #     uniform_samples[inside_mask], dtype=torch.float32, device=self.device
            # )
            # self.orig_vertices = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=self.device)
        elif self.base_type == "gs":
            SH_C0 = 0.28209479177387814
            plydata = PlyData.read(str(self.original_geometry))
            v = plydata["vertex"]

            with torch.no_grad():
                self.orig_positions = torch.tensor(
                    np.stack([v["x"], v["y"], v["z"]], axis=-1), dtype=torch.float32
                ).to(self.device)
                self.orig_positions -= self.orig_positions.mean(dim=0)

                self.scales = torch.tensor(
                    np.exp(
                        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
                    ),
                    dtype=torch.float32,
                ).to(self.device)

                self.wxyzs = torch.tensor(
                    np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
                    dtype=torch.float32,
                ).to(self.device)

                self.colors = 0.5 + SH_C0 * np.stack(
                    [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1
                )
                self.opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

                opacity_threshold = compute_adaptive_opacity_threshold(
                    torch.tensor(self.opacities, device=self.device), self.scales
                )
                self.sim_points = kal.ops.gaussian.sample_points_in_volume(
                    xyz=self.orig_positions,
                    scale=self.scales,
                    rotation=self.wxyzs,
                    opacity=torch.tensor(self.opacities, device=self.device),
                    opacity_threshold=opacity_threshold,
                    jitter=True,
                    clip_samples_to_input_bbox=True,
                )
                pts = self.sim_points
        elif self.base_type == "nerf":
            config, pipeline, checkpoint_path, step = eval_setup(self.original_geometry)
            assert not isinstance(
                pipeline.model, SplatfactoModel
            ), "Base Type is NeRF, but model is SplatFactoModel"

            pipeline.eval()
            pipeline.to(self.device)

            scene_box = pipeline.datamanager.train_dataset.scene_box
            aabb = scene_box.aabb
            bounds = torch.stack([aabb[0], aabb[1]]).cpu().numpy()

            grid_size = 32
            x = torch.linspace(
                bounds[0, 0], bounds[1, 0], grid_size, device=self.device
            )
            y = torch.linspace(
                bounds[0, 1], bounds[1, 1], grid_size, device=self.device
            )
            z = torch.linspace(
                bounds[0, 2], bounds[1, 2], grid_size, device=self.device
            )
            grid_points = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
            grid_points = grid_points.reshape(-1, 3)

            with torch.no_grad():
                grid_density = pipeline.model.field.density_fn(grid_points)
                if isinstance(grid_density, tuple):
                    grid_density = grid_density[0]
                grid_density = grid_density.squeeze(-1)

            if self.density_threshold is None:
                sorted_densities = torch.sort(grid_density, descending=True)[0]
                target_volume_points = int(len(grid_density) * 0.1)
                self.density_threshold = sorted_densities[target_volume_points].item()

            high_density_mask = grid_density > self.density_threshold
            high_density_points = grid_points[high_density_mask]

            if len(high_density_points) == 0:
                raise ValueError(
                    "No high density regions found. Try lowering the density threshold."
                )

            mean = high_density_points.mean(dim=0)
            centered = high_density_points - mean
            cov = torch.mm(centered.T, centered) / (len(high_density_points) - 1)
            cov += torch.eye(3, device=self.device) * 0.01

            collected_points = []
            collected_densities = []

            while len(collected_points) < self.num_samples:
                if len(collected_points) == 0:
                    current_batch_size = self.num_samples * 2
                else:
                    acceptance_rate = len(collected_points) / sum(
                        1 for _ in collected_points
                    )
                    current_batch_size = int(
                        min(
                            (self.num_samples - len(collected_points))
                            / max(acceptance_rate, 0.01),
                            100000,
                        )
                    )

                mix_prob = 0.8
                num_gaussian = int(current_batch_size * mix_prob)
                num_uniform = current_batch_size - num_gaussian

                gaussian_samples = torch.randn(num_gaussian, 3, device=self.device)
                L = torch.linalg.cholesky(cov)
                gaussian_samples = gaussian_samples @ L.T + mean

                bounds = torch.stack([aabb[0], aabb[1]])
                bounds = bounds.to(self.device)

                uniform_samples = torch.rand(num_uniform, 3, device=self.device)
                uniform_samples = uniform_samples * (bounds[1] - bounds[0]) + bounds[0]
                uniform_samples = torch.tensor(
                    uniform_samples, dtype=torch.float32, device=self.device
                )

                samples = torch.cat([gaussian_samples, uniform_samples], dim=0)
                with torch.no_grad():
                    density = pipeline.model.field.density_fn(samples)
                    if isinstance(density, tuple):
                        density = density[0]
                    density = density.squeeze(-1)
                mask = density > self.density_threshold
                valid_points = samples[mask]
                valid_densities = density[mask]
                if len(valid_points) > 0:
                    collected_points.append(valid_points)
                    collected_densities.append(valid_densities)

                total_points = sum(len(p) for p in collected_points)
                if total_points >= self.num_samples:
                    pts = torch.cat(collected_points, dim=0)
                    densities = torch.cat(collected_densities, dim=0)
                    if len(pts) > self.num_samples:
                        idx = torch.randperm(len(pts), device=self.device)[
                            : self.num_samples
                        ]
                        pts = pts[idx]
                        densities = densities[idx]
                    break
            pts, densities = remove_outliers(None, pts, densities)
            self.orig_positions = pts

        yms = torch.full((pts.shape[0],), self.soft_youngs_modulus, device=self.device)
        prs = torch.full((pts.shape[0],), self.poisson_ratio, device=self.device)
        rhos = torch.full((pts.shape[0],), self.rho, device=self.device)

        self.sim_obj = kal.physics.simplicits.SimplicitsObject(
            pts=pts,
            yms=yms,
            prs=prs,
            rhos=rhos,
            appx_vol=torch.tensor(
                [self.approx_volume], dtype=torch.float32, device=self.device
            ),
            num_handles=self.handles,
        )

        self.model = MLP(
            input_size=3,
            hidden_size=64,
            output_size=self.handles,
            num_layers=self.layers,
        )
        self.model = load_model(self.model, self.model_path).to(self.device)
        self.sim_obj.model = self.model

        self.scene = kal.physics.simplicits.SimplicitsScene()
        self.scene.max_newton_steps = 3
        self.obj_idx = self.scene.add_object(self.sim_obj, num_cub_pts=2048)
        self.scene.set_scene_gravity(
            acc_gravity=torch.tensor(self.gravity, device=self.device)
        )
        self.scene.set_scene_floor(
            floor_height=self.floor_height,
            floor_axis=1,
            floor_penalty=self.floor_penalty,
        )

    def simulate_and_cache_frames(self) -> None:
        self.scene.reset()
        self.cached_frames = []
        self.cached_tracks = []

        if self.base_type == "mesh":
            num_vertices = len(self.mesh.vertices)
            sample_rate = max(num_vertices // 100, 1)
            sampled_indices = list(range(0, num_vertices, sample_rate))

            self.mesh.vertices = self.orig_vertices.clone()
            self.cached_frames.append(self.mesh.vertices.clone())

            track_history = [[] for _ in sampled_indices]
            for idx in range(len(sampled_indices)):
                track_history[idx].append(
                    self.orig_vertices[sampled_indices[idx]].clone()
                )
            self.cached_tracks.append([])

            for step in range(self.num_steps):
                self.scene.run_sim_step()
                deformed_vertices = self.scene.get_object_deformed_pts(
                    self.obj_idx, self.orig_vertices
                ).squeeze()
                self.cached_frames.append(deformed_vertices.clone())

                current_tracks = []
                for idx, vert_idx in enumerate(sampled_indices):
                    track_history[idx].append(deformed_vertices[vert_idx].clone())
                    history_start = max(0, len(track_history[idx]) - 5)
                    if len(track_history[idx]) > 1:
                        current_tracks.append(
                            torch.stack(track_history[idx][history_start:])
                        )

                self.cached_tracks.append(current_tracks)
        elif self.base_type == "gs":
            identity_F = torch.eye(3, device=self.device).expand(
                self.orig_positions.shape[0], 3, 3
            )
            self.cached_frames.append([self.orig_positions.clone(), identity_F])

            num_vertices = len(self.orig_positions)
            sample_rate = max(num_vertices // 100, 1)
            sampled_indices = list(range(0, num_vertices, sample_rate))
            track_history = [[] for _ in sampled_indices]
            for idx in range(len(sampled_indices)):
                track_history[idx].append(
                    self.orig_positions[sampled_indices[idx]].clone()
                )
            self.cached_tracks.append([])

            for step in range(self.num_steps):
                self.scene.run_sim_step()

                with torch.no_grad():
                    deformed_positions = self.scene.get_object_deformed_pts(
                        self.obj_idx, self.orig_positions
                    ).squeeze()
                    deformation_grad = self.scene.get_object_deformation_gradient(
                        self.obj_idx, self.orig_positions
                    ).squeeze()

                    self.cached_frames.append(
                        [deformed_positions.clone(), deformation_grad.clone()]
                    )

                    current_tracks = []
                    for idx, vert_idx in enumerate(sampled_indices):
                        track_history[idx].append(deformed_positions[vert_idx].clone())
                        history_start = max(0, len(track_history[idx]) - 5)
                        if len(track_history[idx]) > 1:
                            current_tracks.append(
                                torch.stack(track_history[idx][history_start:])
                            )
                    self.cached_tracks.append(current_tracks)

                if step % 10 == 0:
                    torch.cuda.empty_cache()
        elif self.base_type == "nerf":
            identity_F = torch.eye(3, device=self.device).expand(
                self.orig_positions.shape[0], 3, 3
            )
            self.cached_frames.append([self.orig_positions.clone(), identity_F])

            num_vertices = len(self.orig_positions)
            sample_rate = max(num_vertices // 100, 1)
            sampled_indices = list(range(0, num_vertices, sample_rate))
            track_history = [[] for _ in sampled_indices]
            for idx in range(len(sampled_indices)):
                track_history[idx].append(
                    self.orig_positions[sampled_indices[idx]].clone()
                )
            self.cached_tracks.append([])

            for step in range(self.num_steps):
                self.scene.run_sim_step()

                with torch.no_grad():
                    deformed_positions = self.scene.get_object_deformed_pts(
                        self.obj_idx, self.orig_positions
                    ).squeeze()
                    deformation_grad = self.scene.get_object_deformation_gradient(
                        self.obj_idx, self.orig_positions
                    ).squeeze()

                    self.cached_frames.append(
                        [deformed_positions.clone(), deformation_grad.clone()]
                    )

                    current_tracks = []
                    for idx, vert_idx in enumerate(sampled_indices):
                        track_history[idx].append(deformed_positions[vert_idx].clone())
                        history_start = max(0, len(track_history[idx]) - 5)
                        if len(track_history[idx]) > 1:
                            current_tracks.append(
                                torch.stack(track_history[idx][history_start:])
                            )
                    self.cached_tracks.append(current_tracks)

                if step % 10 == 0:
                    torch.cuda.empty_cache()

    def render_fn_mesh(
        self, camera_state: CameraState, img_wh: tuple[int, int]
    ) -> np.ndarray:
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        if (
            hasattr(self.viewer, "_canonical_checkbox")
            and self.viewer._canonical_checkbox.value
        ):
            frame_idx = 0
        else:
            frame_idx = int(self.viewer._playback_guis[0].value)

        if frame_idx < len(self.cached_frames):
            self.mesh.vertices = self.cached_frames[frame_idx]

        c2w = torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        ray_origins = c2w[:3, 3]
        ray_forward = c2w[:3, 2]
        ray_up = c2w[:3, 1]

        vertices = self.mesh.vertices - ray_origins
        focal = H / (2 * np.tan(camera_state.fov / 2))

        # project to camera space
        cam_z = torch.sum(vertices * ray_forward, dim=-1)
        cam_x = torch.sum(vertices * torch.cross(ray_up, ray_forward), dim=-1)
        cam_y = torch.sum(vertices * ray_up, dim=-1)

        # project to screen space
        px = ((cam_x / cam_z) * focal + W / 2).long()
        py = ((cam_y / cam_z) * focal + H / 2).long()

        img = torch.ones((H, W, 3), device=self.device)

        if (
            hasattr(self.viewer, "_point_checkbox")
            and self.viewer._point_checkbox.value
        ):
            if hasattr(self, "mesh_handle"):
                self.mesh_handle.remove()
            valid_mask = (cam_z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            valid_px = px[valid_mask]
            valid_py = py[valid_mask]

            if valid_px.numel() > 0:
                img[valid_py, valid_px] = torch.tensor(
                    [0.5, 0.5, 0.5], device=self.device
                )
        else:
            self.mesh_handle = self.server.add_glb(
                name=f"dynamic_mesh",
                glb_data=kaolin_to_glb_binary(self.mesh),
                scale=1.0,
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )

        if (
            hasattr(self.viewer, "_render_track_checkbox")
            and self.viewer._render_track_checkbox.value
            and frame_idx < len(self.cached_tracks)
        ):
            for track_points in self.cached_tracks[frame_idx]:
                track_vertices = track_points - ray_origins
                track_z = torch.sum(track_vertices * ray_forward, dim=-1)
                track_x = torch.sum(
                    track_vertices * torch.cross(ray_up, ray_forward), dim=-1
                )
                track_y = torch.sum(track_vertices * ray_up, dim=-1)

                track_px = ((track_x / track_z) * focal + W / 2).long()
                track_py = ((track_y / track_z) * focal + H / 2).long()

                track_valid = (
                    (track_z > 0)
                    & (track_px >= 0)
                    & (track_px < W)
                    & (track_py >= 0)
                    & (track_py < H)
                )

                for i in range(len(track_points) - 1):
                    if track_valid[i] and track_valid[i + 1]:
                        x1, y1 = track_px[i], track_py[i]
                        x2, y2 = track_px[i + 1], track_py[i + 1]

                        steps = max(abs(x2 - x1), abs(y2 - y1))
                        if steps > 0:
                            for t in range(0, steps + 1, 2):
                                x = int(x1 + (x2 - x1) * t / steps)
                                y = int(y1 + (y2 - y1) * t / steps)
                                if 0 <= x < W and 0 <= y < H:
                                    img[y, x] = torch.tensor(
                                        [0.0, 0.8, 0.8], device=self.device
                                    )

        final = (img * 255).to(torch.uint8).cpu().numpy()
        return final

    def render_fn_gs(
        self, camera_state: CameraState, img_wh: tuple[int, int]
    ) -> np.ndarray:
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        if (
            hasattr(self.viewer, "_canonical_checkbox")
            and self.viewer._canonical_checkbox.value
        ):
            frame_idx = 0
        else:
            frame_idx = int(self.viewer._playback_guis[0].value)

        if frame_idx < len(self.cached_frames):
            curr_positions, curr_F = self.cached_frames[frame_idx]

            with torch.no_grad():
                L = build_scaling_rotation(self.scales, self.wxyzs, device=self.device)
                FL = torch.bmm(curr_F, L)
                covariances = FL @ FL.transpose(1, 2)
                identity = torch.eye(3, device=covariances.device)[None].expand(
                    covariances.shape[0], -1, -1
                )
                covariances = (covariances + 1e-5 * identity).to(torch.float32)

                c2w = torch.from_numpy(camera_state.c2w.astype(np.float32)).to(
                    self.device
                )
                focal = float(H / (2 * np.tan(camera_state.fov / 2)))

                intrinsics = torch.tensor(
                    [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0)

                w2c = torch.linalg.inv(c2w)
                viewmat = w2c.unsqueeze(0).to(torch.float32)

                if (
                    hasattr(self.viewer, "_point_checkbox")
                    and self.viewer._point_checkbox.value
                ):
                    ray_origins = c2w[:3, 3]
                    ray_forward = c2w[:3, 2]
                    ray_up = c2w[:3, 1]

                    vertices = curr_positions - ray_origins
                    focal = H / (2 * np.tan(camera_state.fov / 2))

                    # project to camera space
                    cam_z = torch.sum(vertices * ray_forward, dim=-1)
                    cam_x = torch.sum(
                        vertices * torch.cross(ray_up, ray_forward), dim=-1
                    )
                    cam_y = torch.sum(vertices * ray_up, dim=-1)

                    # project to screen space
                    px = ((cam_x / cam_z) * focal + W / 2).long()
                    py = ((cam_y / cam_z) * focal + H / 2).long()

                    img = torch.ones((H, W, 3), device=self.device)
                    valid_mask = (
                        (cam_z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
                    )
                    valid_px = px[valid_mask]
                    valid_py = py[valid_mask]

                    if valid_px.numel() > 0:
                        img[valid_py, valid_px] = torch.tensor(
                            [0.5, 0.5, 0.5], device=self.device
                        )

                    if (
                        hasattr(self.viewer, "_render_track_checkbox")
                        and self.viewer._render_track_checkbox.value
                    ):
                        for track_points in self.cached_tracks[frame_idx]:
                            track_vertices = track_points - ray_origins
                            track_z = torch.sum(track_vertices * ray_forward, dim=-1)
                            track_x = torch.sum(
                                track_vertices * torch.cross(ray_up, ray_forward),
                                dim=-1,
                            )
                            track_y = torch.sum(track_vertices * ray_up, dim=-1)

                            track_px = ((track_x / track_z) * focal + W / 2).long()
                            track_py = ((track_y / track_z) * focal + H / 2).long()

                            track_valid = (
                                (track_z > 0)
                                & (track_px >= 0)
                                & (track_px < W)
                                & (track_py >= 0)
                                & (track_py < H)
                            )

                            for i in range(len(track_points) - 1):
                                if track_valid[i] and track_valid[i + 1]:
                                    x1, y1 = track_px[i], track_py[i]
                                    x2, y2 = track_px[i + 1], track_py[i + 1]

                                    steps = max(abs(x2 - x1), abs(y2 - y1))
                                    if steps > 0:
                                        for t in range(0, steps + 1, 2):
                                            x = int(x1 + (x2 - x1) * t / steps)
                                            y = int(y1 + (y2 - y1) * t / steps)
                                            if 0 <= x < W and 0 <= y < H:
                                                img[y, x] = torch.tensor(
                                                    [0.0, 0.8, 0.8], device=self.device
                                                )
                    return (img * 255).to(torch.uint8).cpu().numpy()

                colors = torch.tensor(
                    self.colors, device=self.device, dtype=torch.float32
                )
                opacities = torch.tensor(
                    self.opacities, device=self.device, dtype=torch.float32
                ).squeeze()
                quats = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]] * len(curr_positions),
                    device=self.device,
                    dtype=torch.float32,
                )
                backgrounds = torch.ones(
                    (1, 3), device=self.device, dtype=torch.float32
                )

                rendered_image, alphas, _ = gs.rasterization(
                    means=curr_positions.to(torch.float32),
                    covars=covariances,
                    quats=quats,
                    scales=self.scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmat,
                    Ks=intrinsics,
                    width=W,
                    height=H,
                    backgrounds=backgrounds,
                )

                rendered_image = (rendered_image[0] * 255).clamp(0, 255).to(torch.uint8)
                return rendered_image.cpu().numpy()

        return np.full((H, W, 3), 255, dtype=np.uint8)

    def render_fn_nerf(
        self, camera_state: CameraState, img_wh: tuple[int, int]
    ) -> np.ndarray:
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        if (
            hasattr(self.viewer, "_canonical_checkbox")
            and self.viewer._canonical_checkbox.value
        ):
            frame_idx = 0
        else:
            frame_idx = int(self.viewer._playback_guis[0].value)

        if frame_idx < len(self.cached_frames):
            curr_positions, curr_F = self.cached_frames[frame_idx]

            c2w = torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
            ray_origins = c2w[:3, 3]
            ray_forward = c2w[:3, 2]
            ray_up = c2w[:3, 1]

            vertices = curr_positions - ray_origins
            focal = H / (2 * np.tan(camera_state.fov / 2))

            # Project to camera space
            cam_z = torch.sum(vertices * ray_forward, dim=-1)
            cam_x = torch.sum(vertices * torch.cross(ray_up, ray_forward), dim=-1)
            cam_y = torch.sum(vertices * ray_up, dim=-1)

            # Project to screen space
            px = ((cam_x / cam_z) * focal + W / 2).long()
            py = ((cam_y / cam_z) * focal + H / 2).long()

            img = torch.ones((H, W, 3), device=self.device)
            valid_mask = (cam_z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            valid_px = px[valid_mask]
            valid_py = py[valid_mask]

            if valid_px.numel() > 0:
                img[valid_py, valid_px] = torch.tensor(
                    [0.5, 0.5, 0.5], device=self.device
                )

            if (
                hasattr(self.viewer, "_render_track_checkbox")
                and self.viewer._render_track_checkbox.value
            ):
                for track_points in self.cached_tracks[frame_idx]:
                    track_vertices = track_points - ray_origins
                    track_z = torch.sum(track_vertices * ray_forward, dim=-1)
                    track_x = torch.sum(
                        track_vertices * torch.cross(ray_up, ray_forward),
                        dim=-1,
                    )
                    track_y = torch.sum(track_vertices * ray_up, dim=-1)

                    track_px = ((track_x / track_z) * focal + W / 2).long()
                    track_py = ((track_y / track_z) * focal + H / 2).long()

                    track_valid = (
                        (track_z > 0)
                        & (track_px >= 0)
                        & (track_px < W)
                        & (track_py >= 0)
                        & (track_py < H)
                    )

                    for i in range(len(track_points) - 1):
                        if track_valid[i] and track_valid[i + 1]:
                            x1, y1 = track_px[i], track_py[i]
                            x2, y2 = track_px[i + 1], track_py[i + 1]

                            steps = max(abs(x2 - x1), abs(y2 - y1))
                            if steps > 0:
                                for t in range(0, steps + 1, 2):
                                    x = int(x1 + (x2 - x1) * t / steps)
                                    y = int(y1 + (y2 - y1) * t / steps)
                                    if 0 <= x < W and 0 <= y < H:
                                        img[y, x] = torch.tensor(
                                            [0.0, 0.8, 0.8], device=self.device
                                        )

            return (img * 255).to(torch.uint8).cpu().numpy()

        return np.full((H, W, 3), 255, dtype=np.uint8)

    def reset_simulation(self) -> None:
        self.scene.reset()
        if self.base_type == "mesh":
            self.mesh.vertices = self.orig_vertices.clone()
        self.cached_frames = []
        self.cached_tracks = []


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    viewer = tyro.cli(RunViewer).main()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    entrypoint()


# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RunViewer)
