import random
from io import BytesIO

import numpy as np
import rich
import torch
import torch.nn.functional as F
import trimesh
from sklearn.neighbors import NearestNeighbors


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def build_covariance_from_scaling_rotation_deformations(
    scaling, scaling_modifier, rotation, defo_grad=None, device="cuda"
):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation, device=device)
    if defo_grad == None:
        FL = L
    else:
        FL = torch.bmm(defo_grad, L)
    actual_covariance = FL @ FL.transpose(1, 2)
    symm = strip_symmetric(actual_covariance, device=device)
    return symm


def kaolin_to_glb_binary(kaolin_mesh, texture_image=None):
    vertices = kaolin_mesh.vertices.detach().cpu().numpy()
    faces = kaolin_mesh.faces.detach().cpu().numpy()

    scene = trimesh.Scene()

    if hasattr(kaolin_mesh, "uvs") and texture_image is not None:
        uvs = kaolin_mesh.uvs.detach().cpu().numpy()
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=trimesh.visual.TextureVisuals(uv=uvs, image=texture_image),
        )
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    scene.add_geometry(mesh)

    glb_buf = BytesIO()
    scene.export(file_obj=glb_buf, file_type="glb")

    return glb_buf.getvalue()


def kaolin_to_trimesh(kaolin_mesh):
    vertices = kaolin_mesh.vertices.detach().cpu().numpy()
    faces = kaolin_mesh.faces.detach().cpu().numpy()

    if hasattr(kaolin_mesh, "uvs") and hasattr(kaolin_mesh, "texture"):
        uvs = kaolin_mesh.uvs.detach().detach().cpu().numpy()
        texture = kaolin_mesh.texture.detach().detach().cpu().numpy()
        vertex_colors = np.zeros((len(vertices), 3))
        for face_idx, face in enumerate(faces):
            for vertex_idx, vert_idx in enumerate(face):
                uv = uvs[face_idx][vertex_idx]
                u, v = uv
                tx = min(int(u * texture.shape[1]), texture.shape[1] - 1)
                ty = min(int(v * texture.shape[0]), texture.shape[0] - 1)
                vertex_colors[vert_idx] = texture[ty, tx]
    else:
        vertex_colors = np.ones((len(vertices), 3)) * 0.5

    # Create trimesh mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=(vertex_colors * 255).astype(np.uint8),  # Convert to 0-255 range
    )

    return mesh


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

    rot = torch.stack(
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * w * z,
            2 * x * z + 2 * w * y,
            2 * x * y + 2 * w * z,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * w * x,
            2 * x * z - 2 * w * y,
            2 * y * z + 2 * w * x,
            1 - 2 * x * x - 2 * y * y,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    return rot


def strip_lowerdiag(L, device):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym, device):
    return strip_lowerdiag(sym, device)


def build_rotation(r, device):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r, device):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=device)
    R = build_rotation(r, device)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def sample_gaussians(
    means,
    scales,
    rotations,
    opacities,
    base_num_points=2048,
    target_density=10,
    device="cuda",
):
    """Adaptively sample points from gaussians.
    Args:
        means: (N, 3) gaussian centers
        scales: (N, 3) gaussian scales (already exp transformed)
        rotations: (N, 4) gaussian rotations as quaternions [w,x,y,z]
        opacities: (N, 1) gaussian opacities (already sigmoid transformed)
        base_num_points: Minimum total points to generate
        target_density: Points per unit volume
        device: Device to use for computations
    """
    num_gaussians = means.shape[0]
    scale_volumes = scales.prod(dim=-1, keepdim=True)
    boxes = []
    total_volume = 0

    for i in range(num_gaussians):
        q = rotations[i]
        R = torch.zeros((3, 3), device=device)

        R[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
        R[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
        R[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
        R[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
        R[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
        R[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
        R[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
        R[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
        R[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2

        corners = torch.tensor(
            [
                [-2, -2, -2],
                [-2, -2, 2],
                [-2, 2, -2],
                [-2, 2, 2],
                [2, -2, -2],
                [2, -2, 2],
                [2, 2, -2],
                [2, 2, 2],
            ],
            device=device,
            dtype=torch.float32,
        )

        corners = corners * scales[i]
        corners = torch.matmul(corners, R.T)
        corners = corners + means[i]
        min_corner = corners.min(dim=0)[0]
        max_corner = corners.max(dim=0)[0]
        box_size = max_corner - min_corner
        box_volume = box_size.prod()
        total_volume += box_volume

        boxes.append(
            {
                "min": min_corner,
                "max": max_corner,
                "volume": box_volume,
                "center": means[i],
                "scale": scales[i],
                "rotation": R,
            }
        )

    points_per_volume = max(base_num_points / total_volume, target_density)
    points_list = []

    for i in range(num_gaussians):
        box = boxes[i]
        opacity_weight = opacities[i].item()
        volume_weight = box["volume"] / total_volume
        importance = opacity_weight * volume_weight
        num_points = int(points_per_volume * box["volume"] * importance)
        num_points = max(num_points, 10)

        points = torch.rand(num_points, 3, device=device) * 2 - 1
        points = torch.erfinv(points) * torch.sqrt(torch.tensor(2.0, device=device))
        points = points * box["scale"]
        points = torch.matmul(points, box["rotation"].T)
        points = points + box["center"]
        jitter = torch.randn_like(points) * min(scales[i]) * 0.05
        points = points + jitter
        points_list.append(points)

    all_points = torch.cat(points_list, dim=0)
    if all_points.shape[0] > base_num_points:
        idx = torch.randperm(all_points.shape[0], device=device)[:base_num_points]
        all_points = all_points[idx]

    return all_points


def compute_adaptive_opacity_threshold(
    opacities: torch.Tensor,
    scales: torch.Tensor,
    target_coverage: float = 0.98,
    min_threshold: float = 0.005,
    max_threshold: float = 0.6,
) -> float:
    """Compute adaptive opacity threshold based on scene characteristics.

    Args:
        opacities: Tensor of shape [N, 1] containing sigmoid-activated opacity values
        scales: Tensor of shape [N, 3] containing exp-activated scale values
        target_coverage: Target fraction of accumulated opacity to preserve
        min_threshold: Minimum allowed opacity threshold
        max_threshold: Maximum allowed opacity threshold

    Returns:
        float: Computed opacity threshold
    """
    opacities_cpu = opacities.cpu()
    scales_cpu = scales.cpu()

    gaussian_volumes = scales_cpu.prod(dim=-1, keepdim=True)

    sorted_idxs = torch.argsort(opacities_cpu.squeeze(), descending=True)
    sorted_opacities = opacities_cpu[sorted_idxs]
    sorted_volumes = gaussian_volumes[sorted_idxs]

    importance = sorted_opacities * sorted_volumes
    cumulative_importance = torch.cumsum(importance, dim=0)
    total_importance = cumulative_importance[-1]

    target_importance = total_importance * target_coverage

    mask = cumulative_importance.squeeze() >= target_importance
    if not mask.any():
        threshold_idx = len(sorted_opacities) - 1
    else:
        threshold_idx = torch.nonzero(mask)[0].item()

    threshold_idx = min(threshold_idx, len(sorted_opacities) - 1)
    initial_threshold = sorted_opacities[threshold_idx].item()

    opacity_mean = opacities_cpu.mean().item()
    opacity_std = opacities_cpu.std().item()

    hist_values = torch.histc(opacities_cpu.squeeze(), bins=100, min=0.0, max=1.0)
    bin_width = 1.0 / 100
    hist_edges = torch.linspace(0, 1, 101)

    total_gaussians = len(opacities)

    hist_diff = hist_values[1:] - hist_values[:-1]
    significant_gaps = torch.nonzero(hist_diff < -total_gaussians * 0.01)

    gap_threshold = float("inf")
    if len(significant_gaps) > 0:
        gap_idx = significant_gaps[0].item()
        gap_threshold = hist_edges[gap_idx + 1].item()

    stat_threshold = opacity_mean + opacity_std
    threshold = min(initial_threshold, stat_threshold, gap_threshold)
    threshold = max(min_threshold, min(threshold, max_threshold))

    return threshold


def remove_outliers(
    CONSOLE: rich.console.Console,
    points: torch.Tensor,
    densities: torch.Tensor,
    k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove outliers from point cloud using multiple criteria.
    Args:
        points: Point cloud tensor (N, 3)
        densities: Density values for each point (N,)
        k: Number of neighbors for local density calculation
    Returns:
        Cleaned points and their corresponding densities
    """
    device = points.device
    N = len(points)

    mean = points.mean(dim=0)
    std = points.std(dim=0)
    zscore = torch.abs((points - mean) / std)
    max_zscore = zscore.max(dim=1)[0]
    statistical_mask = max_zscore < 3.0

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(
        points.cpu().numpy()
    )
    distances, _ = nbrs.kneighbors(points.cpu().numpy())
    avg_distances = torch.tensor(distances[:, 1:].mean(axis=1), device=device)

    distance_mean = avg_distances.mean()
    distance_std = avg_distances.std()
    local_density_mask = avg_distances < (distance_mean + 2 * distance_std)

    density_mean = densities.mean()
    density_std = densities.std()
    density_mask = densities > (density_mean - 2 * density_std)

    final_mask = statistical_mask & local_density_mask & density_mask

    if final_mask.sum() < N * 0.3:
        print("Warning: Too many points would be removed. Adjusting criteria.")
        statistical_mask = zscore.max(dim=1)[0] < 4.0
        local_density_mask = avg_distances < (distance_mean + 3 * distance_std)
        density_mask = densities > (density_mean - 3 * density_std)
        final_mask = statistical_mask & local_density_mask & density_mask

    cleaned_points = points[final_mask]
    cleaned_densities = densities[final_mask]

    if CONSOLE:
        CONSOLE.log(f"Removed {N - len(cleaned_points)} outliers from {N} points")
    return cleaned_points, cleaned_densities
