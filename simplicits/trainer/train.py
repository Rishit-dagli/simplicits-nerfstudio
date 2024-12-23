import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from rich import box, style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from safetensors.torch import load_file, save_file

from .loss import elastic_loss as criterion
from .optimizer import DistributedSimplicitsOptim, SimplicitsOptim
from .scheduler import LinearLRScheduler


def train_simplicits(
    CONSOLE: Console,
    model: nn.Module,
    points: torch.Tensor,
    youngs_modulus: torch.Tensor,
    poisson_ratio: torch.Tensor,
    density: torch.Tensor,
    volume: float,
    num_steps: int = 10000,
    batch_size: int = 10,
    num_samples: int = 1000,
    start_lr: float = 1e-4,  # copied from paper supplementary
    end_lr: float = 1e-5,  # copied from paper supplementary
    save_path: Optional[str] = None,
    verbose: bool = True,
    opt: str = "custom",
    orig_loss: bool = False,
) -> Dict[str, List[float]]:
    """Train a simplicits model.

    Args:
        model: Neural network to train
        points: Input points tensor of shape (num_points, 3)
        youngs_modulus: Young's modulus per point (num_points,)
        poisson_ratio: Poisson ratio per point (num_points,)
        density: Density per point (num_points,)
        volume: Object volume (float)
        num_steps: Number of training steps
        batch_size: Batch size for training
        num_samples: Number of sample points to use per iteration
        start_lr: Initial learning rate
        end_lr: Final learning rate
        le_coeff: Elastic loss coefficient
        save_path: Path to save checkpoint (optional)
        verbose: Should we do logging
        opt: Optimizer to use (custom, adam)
        orig_loss: Should we use the original loss function

    Returns:
        Tuple of (trained model, training metrics dictionary)
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
    )

    if accelerator.is_local_main_process:
        CONSOLE.log(f"Using device: {accelerator.device}")
        CONSOLE.log(f"Saving checkpoints to: {save_path}")

    if opt == "custom":
        optimizer = DistributedSimplicitsOptim(
            model.parameters(), lr=start_lr, ns_steps=6
        )
    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = LinearLRScheduler(optimizer, start_lr, end_lr, num_steps)

    model = accelerator.prepare(model)
    model.train()
    metrics = {"losses": [], "learning_rates": []}

    COLUMN_WIDTHS = [24, 24, 24, 24, 24]
    HEADER = (
        f"{'Step (% Done)':<{COLUMN_WIDTHS[0]}}"
        f"{'Train Iter (time)':<{COLUMN_WIDTHS[1]}}"
        f"{'ETA (time)':<{COLUMN_WIDTHS[2]}}"
        f"{'Loss':<{COLUMN_WIDTHS[4]}}"
        f"{'Train Points / Sec':<{COLUMN_WIDTHS[3]}}"
    )
    CONSOLE.print(HEADER)
    CONSOLE.print("-" * 120)

    start_time = time.time()
    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        sample_indices = torch.randint(
            low=0, high=points.shape[0], size=(num_samples,), device=points.device
        )

        sample_points = points[sample_indices]
        sample_ym = youngs_modulus[sample_indices]
        sample_pr = poisson_ratio[sample_indices]
        sample_rho = density[sample_indices]

        batch_transforms = 0.1 * torch.randn(
            batch_size,
            model.output_size,
            3,
            4,
            dtype=points.dtype,
            device=points.device,
        )

        skinning_weight = model(sample_points)
        elastic_loss = criterion(
            weights=skinning_weight,
            points=sample_points,
            transforms=batch_transforms,
            youngs_modulus=sample_ym,
            poisson_ratio=sample_pr,
            density=sample_rho,
            volume=volume,
            energy_interp=float(step / num_steps),
        )
        if orig_loss:
            elastic_loss = elastic_loss * 1e-1 + 1e6 * torch.nn.MSELoss()(
                skinning_weight.T @ skinning_weight,
                torch.eye(skinning_weight.shape[1], device=skinning_weight.device),
            )

        accelerator.backward(elastic_loss)

        # if accelerator.sync_gradients:
        #     accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        if step % 1000 == 0:
            metrics["losses"].append(elastic_loss)
            metrics["learning_rates"].append(optimizer.param_groups[0]["lr"])

        if verbose and accelerator.is_local_main_process and step % 1000 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_iter_time = elapsed_time / (step + 1)
            eta = avg_iter_time * (num_steps - step)
            train_points_sec = batch_size * num_samples / avg_iter_time / 1e7

            row = [
                f"{step} ({100 * step / num_steps:.2f}%)".ljust(COLUMN_WIDTHS[0]),
                f"{avg_iter_time * 1000:.3f} ms".ljust(COLUMN_WIDTHS[1]),
                f"{int(eta // 60)} m, {int(eta % 60)} s".ljust(COLUMN_WIDTHS[2]),
                f"{elastic_loss:.6f}".ljust(COLUMN_WIDTHS[4]),
                f"{train_points_sec:.2f} M".ljust(COLUMN_WIDTHS[3]),
            ]
            CONSOLE.print(
                f"{row[0]:<20}{row[1]:<20}{row[2]:<20}{row[3]:<20}{row[4]:<20}"
            )

    save_model(
        accelerator.unwrap_model(model), save_path, num_steps, metrics["losses"][-1]
    )

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    table.add_row("Checkpoint Path", str(save_path))
    table.add_row("Execution Time", f"{time.time() - start_time:.2f} s")
    CONSOLE.print(
        Panel(
            table,
            title="[bold][green]:tada: Training Finished :tada:[/bold]",
            expand=False,
        )
    )
    return metrics


def save_model(
    model: nn.Module,
    save_path: str,
    step: int,
    loss: Optional[float] = None,
) -> None:
    """Save model using safetensors format.

    Args:
        model: The model to save
        save_path: Directory to save the model
        step: Current training step
        loss: Optional loss value
    """
    parent_dir = Path(save_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    state_dict = {k: v.cpu() for k, v in state_dict.items()}

    metadata = {
        "step": str(step),
        "loss": str(loss) if loss is not None else "",
        "architecture": model.__class__.__name__,
        "num_handles": str(getattr(model, "num_handles", 0)),
    }

    save_file(state_dict, save_path, metadata=metadata)


def load_model(
    model: nn.Module,
    model_path: str,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model using safetensors format.

    Args:
        model: Model instance to load weights into
        model_path: Path to the saved model file

    Returns:
        Tuple of (loaded model, metadata dictionary)
    """
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model
