from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import kaolin as kal
import numpy as np
import torch
import trimesh
import tyro
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from plyfile import PlyData
from rich.console import Console

from .trainer.model import MLP
from .trainer.train import train_simplicits
from .utils import (
    compute_adaptive_opacity_threshold,
    remove_outliers,
    sample_gaussians,
    seed_all,
)


@dataclass
class Simplicits:

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
    # Path where we need to save the weights
    output_path: Optional[Path] = None
    # Training parameters - device to train on
    device: Optional[str | torch.device] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Training parameters - should we use the original loss function in the
    # paper, i recommend setting this to False
    orig_loss: Optional[bool] = False
    # Training parameters - number of iterations of simplicits
    iters: int = int(1e4)
    # Training parameters - number of points to sample from the geometry
    num_samples: int = int(1e6)
    # Training parameters - number of cubature points to sample, default from
    # paper suppplementary
    cubature_pts: int = 2000
    # Training parameters - number of handles for the model for simulation,
    # default from paper supplementary for popular splats
    handles: int = 40
    # Training parameters - number of layers in the MLP model
    layers: int = 9
    # Training parameters - batch size
    batch_size: int = 16
    # Training parameters - starting learning rate
    start_lr: float = 1e-3
    # Training parameters - ending learning rate
    end_lr: float = 1e-3
    # Training parameters - either "custom" or "adam" (my experiments indicate
    # decreasing quality and decreasing training time from left to right).
    optimizer: str = "custom"
    # Physical material properties - Young's modulus i.e. stiffness
    soft_youngs_modulus: float = 1e5
    # Physical material properties - Poisson's ratio i.e. ratio of lateral
    # strain to longitudinal strain
    poisson_ratio: float = 0.45
    # Physical material properties - Density
    rho: float = 100
    # Physical material properties - Approximate volume of the object
    approx_volume: float = 1

    def main(self) -> None:
        """Main function."""

        seed_all(3407)
        CONSOLE = Console(width=120)

        if self.output_path is None:
            self.output_path = Path("simplicits.safetensors")
        if self.output_path.suffix != ".safetensors":
            self.output_path = self.output_path.with_suffix(".safetensors")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_size = 3
        self.hidden_size = 64  # from paper

        if self.base_type == "mesh":
            mesh = trimesh.load_mesh(
                str(self.original_geometry), force="mesh", process=False
            )
            unique_vertices, _ = np.unique(mesh.vertices, axis=0, return_inverse=True)
            vertices = torch.tensor(unique_vertices, dtype=torch.float32)

            vmin = vertices.min(dim=0, keepdim=True)[0]
            vmax = vertices.max(dim=0, keepdim=True)[0]
            vmid = (vmin + vmax) / 2
            centered_vertices = vertices - vmid
            den = (vmax - vmin).max().clip(min=1e-6)
            centered_vertices = centered_vertices / den

            bounds = mesh.bounds
            uniform_samples = np.random.uniform(
                bounds[0], bounds[1], size=(self.num_samples, 3)
            )
            inside_mask = mesh.contains(uniform_samples)
            pts = torch.tensor(
                uniform_samples[inside_mask], dtype=torch.float32, device=self.device
            )
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
                CONSOLE.log(f"Using density threshold: {self.density_threshold}")

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
            pts, densities = remove_outliers(CONSOLE, pts, densities)
            # self.debug_visualize_points(pts, densities)
        elif self.base_type == "gs":
            plydata = PlyData.read(str(self.original_geometry))
            v = plydata["vertex"]

            with torch.no_grad():
                orig_positions = torch.tensor(
                    np.stack([v["x"], v["y"], v["z"]], axis=-1), dtype=torch.float32
                ).to(self.device)
                orig_positions -= orig_positions.mean(dim=0)

                scales = torch.tensor(
                    np.exp(
                        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
                    ),
                    dtype=torch.float32,
                ).to(self.device)

                wxyzs = torch.tensor(
                    np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
                    dtype=torch.float32,
                ).to(self.device)

                opacities = 1.0 / (1.0 + np.exp(-v["opacity"][:, None]))

                if self.density_threshold is None:
                    opacity_threshold = compute_adaptive_opacity_threshold(
                        torch.tensor(opacities, device=self.device), scales
                    )
                    CONSOLE.log(f"Using opacity threshold: {opacity_threshold}")
                    self.density_threshold = opacity_threshold
                pts = kal.ops.gaussian.sample_points_in_volume(
                    xyz=orig_positions,
                    scale=scales,
                    rotation=wxyzs,
                    opacity=torch.tensor(opacities, device=self.device),
                    opacity_threshold=self.density_threshold,
                    jitter=True,
                    clip_samples_to_input_bbox=True,
                )

        yms = torch.full((pts.shape[0],), self.soft_youngs_modulus, device=self.device)
        prs = torch.full((pts.shape[0],), self.poisson_ratio, device=self.device)
        rhos = torch.full((pts.shape[0],), self.rho, device=self.device)
        volume = torch.tensor(
            [self.approx_volume], dtype=torch.float32, device=self.device
        )

        model = MLP(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.handles,
            num_layers=self.layers,
        )

        train_simplicits(
            CONSOLE=CONSOLE,
            model=model,
            points=pts,
            youngs_modulus=yms,
            poisson_ratio=prs,
            density=rhos,
            volume=volume.item(),
            num_steps=self.iters,
            batch_size=self.batch_size,
            num_samples=self.cubature_pts,
            start_lr=self.start_lr,
            end_lr=self.end_lr,
            save_path=str(self.output_path),
            opt=self.optimizer,
            orig_loss=self.orig_loss,
        )

    def debug_visualize_points(
        self, pts: torch.Tensor, density: torch.Tensor = None, stage: str = ""
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        points = pts.cpu().numpy()
        views = [
            (0, 0),  # Front view
            (90, 0),  # Side view
            (0, 90),  # Top view
            (45, 45),  # 3/4 view
        ]

        fig = plt.figure(figsize=(20, 15))
        for idx, (elev, azim) in enumerate(views, 1):
            ax = fig.add_subplot(2, 2, idx, projection="3d")

            if density is not None:
                density_values = density.cpu().numpy()
                scatter = ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=density_values,
                    cmap="viridis",
                    s=1,
                )
                if idx == 1:
                    plt.colorbar(scatter, label="Density", ax=ax)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="blue", s=1)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            view_name = {
                (0, 0): "Front",
                (90, 0): "Side",
                (0, 90): "Top",
                (45, 45): "3/4",
            }[(elev, azim)]
            ax.set_title(f"{view_name} View")

        plt.suptitle(
            f"Point Cloud Visualization - {stage}\nTotal Points: {len(points)}"
        )
        plt.tight_layout()
        filename = f'debug_points_{stage.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Simplicits).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Simplicits)
