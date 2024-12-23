import einops
import torch


def linear_elastic_energy(
    defo_grad: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor
) -> torch.Tensor:
    """Calculate linear elastic energy.

    Args:
        defo_grad (torch.Tensor): Deformation gradients (batch, points, 3, 3)
        mu (torch.Tensor): First Lame parameter (points,)
        lam (torch.Tensor): Second Lame parameter (points,)

    Returns:
        torch.Tensor: Energy values (batch, points)
    """
    F_T = einops.rearrange(defo_grad, "... i j -> ... j i")

    I = torch.eye(3, device=defo_grad.device)
    I = einops.repeat(
        I,
        "i j -> batch points i j",
        batch=defo_grad.shape[0],
        points=defo_grad.shape[1],
    )

    eps = 0.5 * (F_T + defo_grad) - I

    # Compute trace of strain (batch, points)
    eps_trace = einops.reduce(
        torch.diagonal(eps, dim1=-2, dim2=-1), "b p d -> b p", reduction="sum"
    )

    # Compute Frobenius inner product of strain with itself (batch, points)
    eps_squared = einops.reduce(eps * eps, "b p i j -> b p", reduction="sum")

    mu = einops.rearrange(mu, "p -> 1 p")
    lam = einops.rearrange(lam, "p -> 1 p")

    return mu * eps_squared + 0.5 * lam * eps_trace * eps_trace


def neohookean_energy(
    defo_grad: torch.Tensor, mu: torch.Tensor, lam: torch.Tensor
) -> torch.Tensor:
    """Calculate Neo-Hookean elastic energy.

    Args:
        defo_grad (torch.Tensor): Deformation gradients (batch, points, 3, 3)
        mu (torch.Tensor): First Lame parameter (points,)
        lam (torch.Tensor): Second Lame parameter (points,)

    Returns:
        torch.Tensor: Energy values (batch, points)
    """
    # Compute first invariant I1 = tr(F^T F)
    F_T = einops.rearrange(defo_grad, "... i j -> ... j i")
    FtF = torch.matmul(F_T, defo_grad)
    I1 = einops.reduce(
        torch.diagonal(FtF, dim1=-2, dim2=-1), "... d -> ...", reduction="sum"
    )

    # Compute determinant (batch, points)
    J = torch.linalg.det(defo_grad)

    mu = einops.rearrange(mu, "p -> 1 p")
    lam = einops.rearrange(lam, "p -> 1 p")

    return (mu / 2) * (I1 - 3.0) + (lam / 2) * (J - 1.0) * (J - 1.0) - mu * (J - 1.0)


def elastic_loss(
    weights: torch.Tensor,
    points: torch.Tensor,
    transforms: torch.Tensor,
    youngs_modulus: torch.Tensor,
    poisson_ratio: torch.Tensor,
    density: torch.Tensor,
    volume: float,
    energy_interp: float = 0.5,
) -> torch.Tensor:
    """Calculate combined elastic energy loss.

    Args:
        weights (torch.Tensor): Skinning weights (num_points, num_handles)
        points (torch.Tensor): Points in space (num_points, 3)
        transforms (torch.Tensor): Transformations (batch_size, num_handles, 3, 4)
        youngs_modulus (torch.Tensor): Young's modulus per point (num_points,)
        poisson_ratio (torch.Tensor): Poisson ratio per point (num_points,)
        density (torch.Tensor): Density per point (num_points,)
        volume (float): Object volume
        energy_interp (float): Interpolation factor between linear (0.0) and neo-hookean (1.0)

    Returns:
        torch.Tensor: Total elastic energy loss (scalar)
    """
    # Convert material parameters to Lame parameters
    mu = youngs_modulus / (2 * (1 + poisson_ratio))
    lam = (youngs_modulus * poisson_ratio) / (
        (1 + poisson_ratio) * (1 - 2 * poisson_ratio)
    )

    # Extract rotation and translation
    R = transforms[..., :3]  # (batch_size, num_handles, 3, 3)
    t = transforms[..., 3:]  # (batch_size, num_handles, 3, 1)

    # Compute weighted transformations
    # (batch_size, num_points, 3, 3)
    weighted_R = torch.einsum("ph,bhij -> bpij", weights, R)
    weighted_t = torch.einsum("ph,bhij -> bpij", weights, t)

    # Compute deformation gradient using finite differences
    eps = 1e-6
    delta = torch.eye(3, device=points.device) * eps  # (3, 3)

    # Compute positions and displacements
    # Expand points: (1, num_points, 3)
    points = points.unsqueeze(0)

    # Calculate displaced points: (3, num_points, 3)
    displaced_points = points + delta.unsqueeze(1)

    # Transform base points and displaced points
    # Base points: (batch_size, num_points, 3)
    x0 = torch.einsum("bpij,npj -> bpi", weighted_R, points) + weighted_t[..., 0]

    # Displaced points: (batch_size, 3, num_points, 3)
    x_displaced = torch.einsum(
        "bpij,npj -> bnpi", weighted_R, displaced_points
    ) + weighted_t[..., 0].unsqueeze(1)

    # Compute deformation gradient
    # (batch_size, num_points, 3, 3)
    defo_grad = (x_displaced - x0.unsqueeze(1)).transpose(1, 2) / eps

    # Calculate energies
    linear_energy = linear_elastic_energy(defo_grad, mu, lam)  # (batch, points)
    neo_energy = neohookean_energy(defo_grad, mu, lam)  # (batch, points)

    # Interpolate between energies (batch, points)
    total_energy = (1 - energy_interp) * linear_energy + energy_interp * neo_energy

    # Average over points and batch, then scale by volume
    return (volume / points.shape[1]) * total_energy.mean()
