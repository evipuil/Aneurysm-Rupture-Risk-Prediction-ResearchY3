# Version 1 source snapshot
"""
pinn_correction_batch.py

Batch PINN correction script that:
1. Loads DeepONet predictions from CSV files in predictions/batch_results
2. Uses original VTP geometry for wall points and reference
3. Trains a PINN to refine velocity fields with physics constraints
4. Computes WSS (Wall Shear Stress), OSI (Oscillatory Shear Index), and von Mises stress
5. Outputs corrected flow fields and hemodynamic parameters

Usage:
    python pinn_correction_batch.py --predictions-dir predictions/batch_results --vtp-dir vtp_data
    python pinn_correction_batch.py --csv-file predictions/batch_results/C0001_cut1.csv --vtp-file vtp_data/C0001_cut1.vtp
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy.spatial import KDTree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista is required. Install with: pip install pyvista")
    sys.exit(1)
# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Blood properties (typical values for blood)
MU = 0.0035  # Dynamic viscosity (Pa·s)
RHO = 1060.0  # Density (kg/m³)
NU = MU / RHO  # Kinematic viscosity (m²/s)

# PINN Training parameters
DEFAULT_EPOCHS = 1000
DEFAULT_LR = 1e-3
DEFAULT_N_INTERIOR = 4096
DEFAULT_N_WALL = 2048
DEFAULT_N_SUP = 4096

# Time parameters for pulsatile flow
T_END = 1.0  # Cardiac cycle duration (s)
NT = 10  # Number of time steps

# Random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def setup_logger(log_dir: Path, name: str = "PINNCorrection") -> logging.Logger:
    """Configure and return a logger."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_dir / "pinn_correction.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# NEURAL NETWORK ARCHITECTURES
class SteadyPINN(nn.Module):
    """
    PINN for steady-state Navier-Stokes correction.
    Input: (x, y, z) -> Output: (u, v, w, p)
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()

        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))  # u, v, w, p

        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnsteadyPINN(nn.Module):
    """
    PINN for unsteady Navier-Stokes correction.
    Input: (x, y, z, t) -> Output: (u, v, w, p)
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()

        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))  # u, v, w, p

        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CorrectionPINN(nn.Module):
    """
    PINN that learns corrections to DeepONet predictions.
    Input: (x, y, z, t) -> Output: (δu, δv, δw, δp)
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()

        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))  # δu, δv, δw, δp

        self.net = nn.Sequential(*layers)

        # Initialize with small weights for small corrections
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# UTILITY FUNCTIONS
def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute gradients of y with respect to x using autograd."""
    return autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def pulsatile_scale(t: torch.Tensor, t_end: float = T_END) -> torch.Tensor:
    """
    Pulsatile flow scaling factor simulating cardiac cycle.
    Returns scale factor between 0.4 and 1.6.
    """
    return 1.0 + 0.6 * torch.sin(2 * np.pi * t / t_end)


def load_deeponet_predictions(csv_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DeepONet predictions from CSV file.

    Returns:
        xyz: (N, 3) coordinates
        velocity: (N, 3) velocity components [u, v, w]
        pressure: (N, 1) pressure
    """
    data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
    xyz = data[:, 0:3].astype(np.float32)
    pressure = data[:, 3:4].astype(np.float32)
    velocity = data[:, 4:7].astype(np.float32)
    return xyz, velocity, pressure


def extract_wall_points(vtp_file: Path, n_wall_points: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract wall points and normals from VTP file.

    Returns:
        wall_points: (N, 3) wall point coordinates
        wall_normals: (N, 3) outward normal vectors
    """
    mesh = pv.read(vtp_file)

    # Get surface if needed
    if isinstance(mesh, pv.UnstructuredGrid):
        surface = mesh.extract_surface()
    else:
        surface = mesh

    # Compute normals if not present
    if surface.point_normals is None:
        surface = surface.compute_normals(point_normals=True, cell_normals=False)

    wall_points = surface.points.astype(np.float32)
    wall_normals = surface.point_normals.astype(np.float32)

    # Subsample if needed
    if len(wall_points) > n_wall_points:
        idx = np.random.choice(len(wall_points), n_wall_points, replace=False)
        wall_points = wall_points[idx]
        wall_normals = wall_normals[idx]

    return wall_points, wall_normals


def compute_wss(
    velocity: torch.Tensor,
    points: torch.Tensor,
    wall_points: torch.Tensor,
    wall_normals: torch.Tensor,
    mu: float = MU,
) -> torch.Tensor:
    """
    Compute Wall Shear Stress (WSS) at wall points.

    WSS = μ * (∂u/∂n) where n is the wall normal direction.
    Using nearest neighbor interpolation and velocity gradient estimation.

    Returns:
        wss_vector: (N_wall, 3) WSS vector at each wall point
    """
    # Find nearest interior points to each wall point
    tree = KDTree(points.detach().cpu().numpy())
    distances, indices = tree.query(wall_points.detach().cpu().numpy(), k=5)

    # Compute velocity at wall (should be ~0 for no-slip, but we use nearby values)
    indices_tensor = torch.tensor(indices, device=velocity.device)

    # Get nearby velocities and estimate gradient
    nearby_vels = velocity[indices_tensor]  # (N_wall, k, 3)

    # Use finite difference approximation for velocity gradient
    # WSS ≈ μ * (v_nearby - v_wall) / distance
    distances_tensor = torch.tensor(distances, device=velocity.device, dtype=torch.float32)
    distances_tensor = distances_tensor.clamp(min=1e-6)

    # Average velocity from nearby points
    avg_vel = nearby_vels.mean(dim=1)  # (N_wall, 3)
    avg_dist = distances_tensor.mean(dim=1, keepdim=True)  # (N_wall, 1)

    # Wall normal unit vectors
    wall_normals_unit = wall_normals / (torch.norm(wall_normals, dim=-1, keepdim=True) + 1e-8)

    # Tangential velocity component (velocity parallel to wall)
    vel_normal = torch.sum(avg_vel * wall_normals_unit, dim=-1, keepdim=True) * wall_normals_unit
    vel_tangent = avg_vel - vel_normal

    # WSS = μ * du_tangent/dn
    wss_vector = mu * vel_tangent / avg_dist

    return wss_vector


def compute_wss_magnitude(wss_vector: torch.Tensor) -> torch.Tensor:
    """Compute WSS magnitude from WSS vector."""
    return torch.norm(wss_vector, dim=-1)


def compute_osi(wss_history: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute Oscillatory Shear Index (OSI) from time history of WSS vectors.

    OSI = 0.5 * (1 - |∫WSS dt| / ∫|WSS| dt)

    OSI ranges from 0 (unidirectional flow) to 0.5 (fully oscillatory).

    Args:
        wss_history: List of WSS vectors at different time points

    Returns:
        osi: (N_wall,) OSI value at each wall point
    """
    # Stack WSS history
    wss_stack = torch.stack(wss_history, dim=0)  # (NT, N_wall, 3)

    # Time-averaged WSS vector
    wss_avg = wss_stack.mean(dim=0)  # (N_wall, 3)

    # Time-averaged WSS magnitude
    wss_mag_avg = torch.norm(wss_stack, dim=-1).mean(dim=0)  # (N_wall,)

    # Magnitude of time-averaged WSS vector
    wss_avg_mag = torch.norm(wss_avg, dim=-1)  # (N_wall,)

    # OSI calculation
    osi = 0.5 * (1.0 - wss_avg_mag / (wss_mag_avg + 1e-8))

    return osi


def compute_von_mises_stress(wss_vector: torch.Tensor) -> torch.Tensor:
    """
    Compute von Mises stress from WSS vector.

    For a 2D stress state on the wall (shear only):
    σ_vm = √(3 * τ²) = √3 * |WSS|

    This is a simplified calculation assuming the wall experiences
    primarily shear stress from the fluid.

    Returns:
        von_mises: (N_wall,) von Mises stress at each wall point
    """
    wss_mag = torch.norm(wss_vector, dim=-1)
    von_mises = np.sqrt(3) * wss_mag
    return von_mises


def compute_tawss(wss_history: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute Time-Averaged Wall Shear Stress (TAWSS).

    TAWSS = (1/T) * ∫|WSS| dt

    Returns:
        tawss: (N_wall,) TAWSS at each wall point
    """
    wss_mags = torch.stack([torch.norm(wss, dim=-1) for wss in wss_history], dim=0)
    tawss = wss_mags.mean(dim=0)
    return tawss


# PINN TRAINING FUNCTIONS
def compute_ns_residuals_steady(
    model: nn.Module, X_int: torch.Tensor, nu: float = NU
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Navier-Stokes residuals for steady flow.

    Returns:
        mom_u, mom_v, mom_w: Momentum residuals
        cont: Continuity residual
    """
    X_int = X_int.requires_grad_(True)
    out = model(X_int)
    u, v, w, p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    # First derivatives
    u_x = gradients(u, X_int)[:, 0:1]
    u_y = gradients(u, X_int)[:, 1:2]
    u_z = gradients(u, X_int)[:, 2:3]

    v_x = gradients(v, X_int)[:, 0:1]
    v_y = gradients(v, X_int)[:, 1:2]
    v_z = gradients(v, X_int)[:, 2:3]

    w_x = gradients(w, X_int)[:, 0:1]
    w_y = gradients(w, X_int)[:, 1:2]
    w_z = gradients(w, X_int)[:, 2:3]

    p_x = gradients(p, X_int)[:, 0:1]
    p_y = gradients(p, X_int)[:, 1:2]
    p_z = gradients(p, X_int)[:, 2:3]

    # Second derivatives (Laplacian)
    u_xx = gradients(u_x, X_int)[:, 0:1]
    u_yy = gradients(u_y, X_int)[:, 1:2]
    u_zz = gradients(u_z, X_int)[:, 2:3]

    v_xx = gradients(v_x, X_int)[:, 0:1]
    v_yy = gradients(v_y, X_int)[:, 1:2]
    v_zz = gradients(v_z, X_int)[:, 2:3]

    w_xx = gradients(w_x, X_int)[:, 0:1]
    w_yy = gradients(w_y, X_int)[:, 1:2]
    w_zz = gradients(w_z, X_int)[:, 2:3]

    # Steady Navier-Stokes: (u·∇)u + ∇p - ν∇²u = 0
    mom_u = (u * u_x + v * u_y + w * u_z) + p_x - nu * (u_xx + u_yy + u_zz)
    mom_v = (u * v_x + v * v_y + w * v_z) + p_y - nu * (v_xx + v_yy + v_zz)
    mom_w = (u * w_x + v * w_y + w * w_z) + p_z - nu * (w_xx + w_yy + w_zz)

    # Continuity: ∇·u = 0
    cont = u_x + v_y + w_z

    return mom_u, mom_v, mom_w, cont


def train_steady_pinn_correction(
    X_int: torch.Tensor,
    Y_int_baseline: torch.Tensor,
    X_sup: torch.Tensor,
    Y_sup_baseline: torch.Tensor,
    X_wall: torch.Tensor,
    Y_wall_baseline: torch.Tensor,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    logger: Optional[logging.Logger] = None,
) -> nn.Module:
    """
    Train steady-state PINN that learns corrections to DeepONet baseline.

    Final prediction = DeepONet baseline + PINN correction
    This approach is more stable and keeps corrections small.

    Args:
        X_int: Interior points for physics loss (N_int, 3)
        Y_int_baseline: DeepONet baseline at interior points [u, v, w, p] (N_int, 4)
        X_sup: Supervision points (N_sup, 3)
        Y_sup_baseline: DeepONet baseline at supervision points (N_sup, 4)
        X_wall: Wall points for no-slip BC (N_wall, 3)
        Y_wall_baseline: DeepONet baseline at wall points (N_wall, 4)

    Returns:
        Trained correction PINN model
    """
    log = logger.info if logger else print

    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=200)

    X_int = X_int.to(DEVICE).requires_grad_(True)
    Y_int_baseline = Y_int_baseline.to(DEVICE)
    X_sup = X_sup.to(DEVICE)
    Y_sup_baseline = Y_sup_baseline.to(DEVICE)
    X_wall = X_wall.to(DEVICE)
    Y_wall_baseline = Y_wall_baseline.to(DEVICE)

    log(
        f"Training steady correction PINN: {epochs} epochs, {X_int.shape[0]} interior pts, {X_wall.shape[0]} wall pts"
    )
    log("Using DeepONet predictions as baseline with learned corrections")

    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="PINN Correction Training")

    for epoch in pbar:
        optimizer.zero_grad()

        # Get corrections at interior points
        # Add dummy time dimension for CorrectionPINN (use t=0 for steady)
        t_zeros_int = torch.zeros((X_int.shape[0], 1), device=DEVICE)
        XT_int = torch.cat([X_int, t_zeros_int], dim=1).requires_grad_(True)

        delta = model(XT_int)
        du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]

        # Corrected field = baseline + correction
        u = Y_int_baseline[:, 0:1] + du
        v = Y_int_baseline[:, 1:2] + dv
        w = Y_int_baseline[:, 2:3] + dw
        p = Y_int_baseline[:, 3:4] + dp

        # Compute Navier-Stokes residuals on corrected field
        u_x = gradients(u, XT_int)[:, 0:1]
        u_y = gradients(u, XT_int)[:, 1:2]
        u_z = gradients(u, XT_int)[:, 2:3]

        v_x = gradients(v, XT_int)[:, 0:1]
        v_y = gradients(v, XT_int)[:, 1:2]
        v_z = gradients(v, XT_int)[:, 2:3]

        w_x = gradients(w, XT_int)[:, 0:1]
        w_y = gradients(w, XT_int)[:, 1:2]
        w_z = gradients(w, XT_int)[:, 2:3]

        p_x = gradients(p, XT_int)[:, 0:1]
        p_y = gradients(p, XT_int)[:, 1:2]
        p_z = gradients(p, XT_int)[:, 2:3]

        u_xx = gradients(u_x, XT_int)[:, 0:1]
        u_yy = gradients(u_y, XT_int)[:, 1:2]
        u_zz = gradients(u_z, XT_int)[:, 2:3]

        v_xx = gradients(v_x, XT_int)[:, 0:1]
        v_yy = gradients(v_y, XT_int)[:, 1:2]
        v_zz = gradients(v_z, XT_int)[:, 2:3]

        w_xx = gradients(w_x, XT_int)[:, 0:1]
        w_yy = gradients(w_y, XT_int)[:, 1:2]
        w_zz = gradients(w_z, XT_int)[:, 2:3]

        # Steady Navier-Stokes residuals
        mom_u = (u * u_x + v * u_y + w * u_z) + p_x - NU * (u_xx + u_yy + u_zz)
        mom_v = (u * v_x + v * v_y + w * v_z) + p_y - NU * (v_xx + v_yy + v_zz)
        mom_w = (u * w_x + v * w_y + w * w_z) + p_z - NU * (w_xx + w_yy + w_zz)
        cont = u_x + v_y + w_z

        loss_physics = (
            mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
        )

        # Wall boundary loss: corrected velocity should be zero (no-slip)
        # baseline + correction = 0, so correction = -baseline at wall
        t_zeros_wall = torch.zeros((X_wall.shape[0], 1), device=DEVICE)
        XT_wall = torch.cat([X_wall, t_zeros_wall], dim=1)
        delta_wall = model(XT_wall)

        # Corrected wall velocity should be zero
        u_wall = Y_wall_baseline[:, 0:1] + delta_wall[:, 0:1]
        v_wall = Y_wall_baseline[:, 1:2] + delta_wall[:, 1:2]
        w_wall = Y_wall_baseline[:, 2:3] + delta_wall[:, 2:3]

        loss_wall = u_wall.pow(2).mean() + v_wall.pow(2).mean() + w_wall.pow(2).mean()

        # Regularization: penalize large corrections
        loss_correction = du.pow(2).mean() + dv.pow(2).mean() + dw.pow(2).mean() + dp.pow(2).mean()

        # Total loss with weighting
        loss = loss_physics + 2.0 * loss_wall + 0.1 * loss_correction

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 100 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3e}",
                    "phys": f"{loss_physics.item():.3e}",
                    "wall": f"{loss_wall.item():.3e}",
                    "corr": f"{loss_correction.item():.3e}",
                }
            )

    log(f"Training complete. Best loss: {best_loss:.3e}")
    return model


def train_unsteady_pinn_correction(
    X_int: torch.Tensor,
    Y_int_baseline: torch.Tensor,
    X_wall: torch.Tensor,
    Y_wall_baseline: torch.Tensor,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    nt: int = NT,
    logger: Optional[logging.Logger] = None,
) -> nn.Module:
    """
    Train unsteady PINN that learns corrections to DeepONet baseline with pulsatile flow.

    Final prediction = pulsatile_scale(t) * DeepONet baseline + PINN correction

    Args:
        X_int: Interior points (N_int, 3)
        Y_int_baseline: DeepONet baseline at interior points [u, v, w, p] (N_int, 4)
        X_wall: Wall points (N_wall, 3)
        Y_wall_baseline: DeepONet baseline at wall points (N_wall, 4)

    Returns:
        Trained correction PINN model
    """
    log = logger.info if logger else print

    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_int = X_int.to(DEVICE)
    Y_int_baseline = Y_int_baseline.to(DEVICE)
    X_wall = X_wall.to(DEVICE)
    Y_wall_baseline = Y_wall_baseline.to(DEVICE)

    t_vals = torch.linspace(0, T_END, nt, device=DEVICE)

    log(f"Training unsteady correction PINN: {epochs} epochs, {nt} time steps")
    log("Using DeepONet predictions as baseline with pulsatile scaling + learned corrections")

    pbar = tqdm(range(epochs), desc="PINN Correction Training")

    for epoch in pbar:
        optimizer.zero_grad()
        loss_physics = 0.0
        loss_wall = 0.0
        loss_correction = 0.0

        for t in t_vals:
            # Add time dimension
            t_col = t * torch.ones((X_int.shape[0], 1), device=DEVICE)
            XT_int = torch.cat([X_int, t_col], dim=1).requires_grad_(True)

            # Pulsatile scaling of baseline
            scale = pulsatile_scale(t)

            # Get corrections
            delta = model(XT_int)
            du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]

            # Corrected field = scaled baseline + correction
            u = scale * Y_int_baseline[:, 0:1] + du
            v = scale * Y_int_baseline[:, 1:2] + dv
            w = scale * Y_int_baseline[:, 2:3] + dw
            p = Y_int_baseline[:, 3:4] + dp  # Pressure not scaled

            # Compute derivatives (including time)
            u_t = gradients(u, XT_int)[:, 3:4]
            v_t = gradients(v, XT_int)[:, 3:4]
            w_t = gradients(w, XT_int)[:, 3:4]

            u_x = gradients(u, XT_int)[:, 0:1]
            u_y = gradients(u, XT_int)[:, 1:2]
            u_z = gradients(u, XT_int)[:, 2:3]

            v_x = gradients(v, XT_int)[:, 0:1]
            v_y = gradients(v, XT_int)[:, 1:2]
            v_z = gradients(v, XT_int)[:, 2:3]

            w_x = gradients(w, XT_int)[:, 0:1]
            w_y = gradients(w, XT_int)[:, 1:2]
            w_z = gradients(w, XT_int)[:, 2:3]

            p_x = gradients(p, XT_int)[:, 0:1]
            p_y = gradients(p, XT_int)[:, 1:2]
            p_z = gradients(p, XT_int)[:, 2:3]

            u_xx = gradients(u_x, XT_int)[:, 0:1]
            u_yy = gradients(u_y, XT_int)[:, 1:2]
            u_zz = gradients(u_z, XT_int)[:, 2:3]

            v_xx = gradients(v_x, XT_int)[:, 0:1]
            v_yy = gradients(v_y, XT_int)[:, 1:2]
            v_zz = gradients(v_z, XT_int)[:, 2:3]

            w_xx = gradients(w_x, XT_int)[:, 0:1]
            w_yy = gradients(w_y, XT_int)[:, 1:2]
            w_zz = gradients(w_z, XT_int)[:, 2:3]

            # Unsteady Navier-Stokes residuals
            mom_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - NU * (u_xx + u_yy + u_zz)
            mom_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - NU * (v_xx + v_yy + v_zz)
            mom_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - NU * (w_xx + w_yy + w_zz)
            cont = u_x + v_y + w_z

            loss_physics += (
                mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
            )

            # Wall BC: corrected velocity should be zero (no-slip)
            t_col_wall = t * torch.ones((X_wall.shape[0], 1), device=DEVICE)
            XT_wall = torch.cat([X_wall, t_col_wall], dim=1)
            delta_wall = model(XT_wall)

            u_wall = scale * Y_wall_baseline[:, 0:1] + delta_wall[:, 0:1]
            v_wall = scale * Y_wall_baseline[:, 1:2] + delta_wall[:, 1:2]
            w_wall = scale * Y_wall_baseline[:, 2:3] + delta_wall[:, 2:3]

            loss_wall += u_wall.pow(2).mean() + v_wall.pow(2).mean() + w_wall.pow(2).mean()

            # Regularization: penalize large corrections
            loss_correction += (
                du.pow(2).mean() + dv.pow(2).mean() + dw.pow(2).mean() + dp.pow(2).mean()
            )

        # Average over time steps
        loss_physics /= nt
        loss_wall /= nt
        loss_correction /= nt

        loss = loss_physics + 2.0 * loss_wall + 0.1 * loss_correction
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3e}",
                    "phys": f"{loss_physics.item():.3e}",
                    "wall": f"{loss_wall.item():.3e}",
                }
            )

    log("Unsteady correction PINN training complete")
    return model


# MAIN PROCESSING FUNCTIONS
def process_single_case(
    csv_file: Path,
    vtp_file: Path,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    n_interior: int = DEFAULT_N_INTERIOR,
    n_wall: int = DEFAULT_N_WALL,
    unsteady: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Process a single case: load data, train PINN, compute hemodynamic parameters.

    Args:
        csv_file: Path to DeepONet prediction CSV
        vtp_file: Path to original VTP geometry
        output_dir: Output directory for results
        epochs: Number of training epochs
        n_interior: Number of interior points for physics
        n_wall: Number of wall points
        unsteady: Whether to use unsteady PINN

    Returns:
        Dictionary with results and timing
    """
    log = logger.info if logger else print

    case_name = csv_file.stem
    start_time = time.time()

    log(f"Processing: {case_name}")

    # Load DeepONet predictions
    log("  Loading DeepONet predictions...")
    xyz, velocity, pressure = load_deeponet_predictions(csv_file)
    n_points = len(xyz)

    # Load wall points from VTP
    log("  Extracting wall points from VTP...")
    wall_points, wall_normals = extract_wall_points(vtp_file, n_wall)

    # Prepare tensors
    X_all = torch.tensor(xyz, dtype=torch.float32, device=DEVICE)
    V_all = torch.tensor(velocity, dtype=torch.float32, device=DEVICE)
    P_all = torch.tensor(pressure, dtype=torch.float32, device=DEVICE)
    Y_all = torch.cat([V_all, P_all], dim=1)  # (N, 4): u, v, w, p

    X_wall = torch.tensor(wall_points, dtype=torch.float32, device=DEVICE)
    N_wall = torch.tensor(wall_normals, dtype=torch.float32, device=DEVICE)

    # Sample interior and supervision points
    n_sup = min(n_points, DEFAULT_N_SUP)
    n_int = min(n_points, n_interior)

    idx_sup = np.random.choice(n_points, n_sup, replace=False)
    idx_int = np.random.choice(n_points, n_int, replace=False)

    X_sup = X_all[idx_sup]
    Y_sup = Y_all[idx_sup]
    X_int = X_all[idx_int]

    # Get baseline predictions at wall points (for correction training)
    # Use KDTree to find nearest interior points to wall points
    tree = KDTree(xyz)
    _, wall_nn_idx = tree.query(wall_points, k=1)
    Y_wall_baseline = torch.tensor(
        np.concatenate([velocity[wall_nn_idx], pressure[wall_nn_idx]], axis=1),
        dtype=torch.float32,
        device=DEVICE,
    )

    # Get baseline at interior points
    Y_int_baseline = Y_all[idx_int]

    # Train PINN with correction approach
    log("  Training correction PINN...")
    train_start = time.time()

    if unsteady:
        model = train_unsteady_pinn_correction(
            X_int, Y_int_baseline, X_wall, Y_wall_baseline, epochs=epochs, logger=logger
        )
    else:
        model = train_steady_pinn_correction(
            X_int,
            Y_int_baseline,
            X_sup,
            Y_sup,
            X_wall,
            Y_wall_baseline,
            epochs=epochs,
            logger=logger,
        )

    train_time = time.time() - train_start
    log(f"  PINN training time: {train_time:.1f}s")

    # Compute corrected velocity field
    log("  Computing corrected velocity field...")
    model.eval()

    with torch.no_grad():
        if unsteady:
            # Use mid-cycle time point for output
            t_mid = T_END / 2
            scale = pulsatile_scale(torch.tensor(t_mid, device=DEVICE))
            t_col = t_mid * torch.ones((X_all.shape[0], 1), device=DEVICE)
            XT_all = torch.cat([X_all, t_col], dim=1)
            delta_all = model(XT_all)

            # Corrected = scaled baseline + correction
            velocity_corrected = (scale * V_all + delta_all[:, :3]).cpu().numpy()
            pressure_corrected = (P_all + delta_all[:, 3:4]).cpu().numpy()
        else:
            t_col = torch.zeros((X_all.shape[0], 1), device=DEVICE)
            XT_all = torch.cat([X_all, t_col], dim=1)
            delta_all = model(XT_all)

            # Corrected = baseline + correction
            velocity_corrected = (V_all + delta_all[:, :3]).cpu().numpy()
            pressure_corrected = (P_all + delta_all[:, 3:4]).cpu().numpy()

    # Compute WSS and other hemodynamic parameters
    log("  Computing hemodynamic parameters...")

    if unsteady:
        # Compute WSS at multiple time points for OSI
        wss_history = []
        t_vals = np.linspace(0, T_END, NT)

        for t in t_vals:
            scale = pulsatile_scale(torch.tensor(t, device=DEVICE))
            t_col = torch.full((X_all.shape[0], 1), t, device=DEVICE)
            XT = torch.cat([X_all, t_col], dim=1)

            with torch.no_grad():
                delta = model(XT)
                # Corrected velocity = scaled baseline + correction
                vel = scale * V_all + delta[:, :3]

            wss = compute_wss(vel, X_all, X_wall, N_wall)
            wss_history.append(wss)

        # Compute time-averaged metrics
        wss_final = wss_history[NT // 2]  # Mid-cycle WSS
        tawss = compute_tawss(wss_history)
        osi = compute_osi(wss_history)
        von_mises = compute_von_mises_stress(wss_final)

    else:
        # Steady-state: single WSS computation
        t_col = torch.zeros((X_all.shape[0], 1), device=DEVICE)
        XT = torch.cat([X_all, t_col], dim=1)

        with torch.no_grad():
            delta = model(XT)
            # Corrected velocity = baseline + correction
            vel = V_all + delta[:, :3]

        wss_final = compute_wss(vel, X_all, X_wall, N_wall)
        wss_mag = compute_wss_magnitude(wss_final)
        von_mises = compute_von_mises_stress(wss_final)
        tawss = wss_mag
        osi = torch.zeros_like(wss_mag)  # No oscillation in steady flow

    # Save results
    log("  Saving results...")

    # Create case-specific folder
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    timesteps_dir = case_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)

    if unsteady:
        # Save time-dependent data for each timestep
        log("  Saving time-dependent fields for each timestep...")
        t_vals = np.linspace(0, T_END, NT)

        for ti, t in enumerate(t_vals):
            scale = pulsatile_scale(torch.tensor(t, device=DEVICE))
            t_col = torch.full((X_all.shape[0], 1), t, device=DEVICE)
            XT = torch.cat([X_all, t_col], dim=1)

            with torch.no_grad():
                delta = model(XT)
                vel_t = scale * V_all + delta[:, :3]
                pres_t = P_all + delta[:, 3:4]

            # Compute WSS at this timestep
            wss_t = compute_wss(vel_t, X_all, X_wall, N_wall)
            wss_mag_t = compute_wss_magnitude(wss_t)

            # Save velocity/pressure field for this timestep
            velocity_t = vel_t.cpu().numpy()
            pressure_t = pres_t.cpu().numpy()

            flow_csv = timesteps_dir / f"flow_t{ti:02d}.csv"
            flow_data = np.concatenate([xyz, pressure_t, velocity_t], axis=1)
            np.savetxt(
                flow_csv,
                flow_data,
                delimiter=",",
                header=f"x,y,z,p,u,v,w,time={t:.4f}s",
                comments="",
            )

            # Save WSS at wall points for this timestep
            wss_csv = timesteps_dir / f"wss_t{ti:02d}.csv"
            wss_data = np.concatenate(
                [wall_points, wss_t.cpu().numpy(), wss_mag_t.cpu().numpy().reshape(-1, 1)], axis=1
            )
            np.savetxt(
                wss_csv,
                wss_data,
                delimiter=",",
                header=f"x,y,z,wss_x,wss_y,wss_z,wss_magnitude,time={t:.4f}s",
                comments="",
            )

            # Save VTP for this timestep
            vtp_t = timesteps_dir / f"flow_t{ti:02d}.vtp"
            wall_mesh_t = pv.PolyData(wall_points)
            wall_mesh_t["WSS"] = wss_mag_t.cpu().numpy()
            wall_mesh_t["WSS_Vector"] = wss_t.cpu().numpy()
            wall_mesh_t.save(vtp_t)

        # Save time index file
        time_index = timesteps_dir / "time_index.csv"
        np.savetxt(
            time_index,
            np.column_stack([np.arange(NT), t_vals]),
            delimiter=",",
            header="timestep,time_seconds",
            comments="",
        )
    else:
        # Steady mode: save single flow field in timesteps folder for consistency
        flow_csv = timesteps_dir / "flow_steady.csv"
        corrected_data = np.concatenate([xyz, pressure_corrected, velocity_corrected], axis=1)
        np.savetxt(flow_csv, corrected_data, delimiter=",", header="x,y,z,p,u,v,w", comments="")

        # Save WSS for steady case
        wss_csv = timesteps_dir / "wss_steady.csv"
        wss_mag = compute_wss_magnitude(wss_final)
        wss_data = np.concatenate(
            [wall_points, wss_final.cpu().numpy(), wss_mag.cpu().numpy().reshape(-1, 1)], axis=1
        )
        np.savetxt(
            wss_csv,
            wss_data,
            delimiter=",",
            header="x,y,z,wss_x,wss_y,wss_z,wss_magnitude",
            comments="",
        )

    # Save aggregate hemodynamic parameters (TAWSS, OSI, von Mises) in main case folder
    log("  Saving aggregate hemodynamic parameters...")
    aggregate_csv = case_dir / "hemodynamics_aggregate.csv"
    aggregate_data = np.concatenate(
        [
            wall_points,
            tawss.cpu().numpy().reshape(-1, 1),
            osi.cpu().numpy().reshape(-1, 1),
            von_mises.cpu().numpy().reshape(-1, 1),
        ],
        axis=1,
    )
    np.savetxt(
        aggregate_csv,
        aggregate_data,
        delimiter=",",
        header="x,y,z,tawss,osi,von_mises",
        comments="",
    )

    # Save PINN model in case folder
    model_file = case_dir / "pinn_model.pt"
    torch.save(model.state_dict(), model_file)

    # Save aggregate VTP with hemodynamic scalars for visualization
    vtp_aggregate = case_dir / "hemodynamics_aggregate.vtp"
    wall_mesh = pv.PolyData(wall_points)
    wall_mesh["TAWSS"] = tawss.cpu().numpy()
    wall_mesh["OSI"] = osi.cpu().numpy()
    wall_mesh["VonMises"] = von_mises.cpu().numpy()
    wall_mesh.save(vtp_aggregate)

    total_time = time.time() - start_time
    log(f"  Completed in {total_time:.1f}s")
    log(f"  Output folder: {case_dir}")

    return {
        "case_name": case_name,
        "n_points": n_points,
        "n_wall": len(wall_points),
        "train_time": train_time,
        "total_time": total_time,
        "output_dir": str(case_dir),
        "output_aggregate": str(aggregate_csv),
        "output_timesteps": str(timesteps_dir),
        "wss_mean": float(compute_wss_magnitude(wss_final).mean().cpu().numpy()),
        "tawss_mean": float(tawss.mean().cpu().numpy()),
        "osi_mean": float(osi.mean().cpu().numpy()),
        "von_mises_mean": float(von_mises.mean().cpu().numpy()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PINN correction for DeepONet predictions with WSS/OSI/VonMises computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all files in batch_results:
    python pinn_correction_batch.py --predictions-dir predictions/batch_results --vtp-dir vtp_data

  Process a single case:
    python pinn_correction_batch.py --csv-file predictions/batch_results/C0001_cut1.csv --vtp-file vtp_data/C0001_cut1.vtp

  Quick test with fewer epochs:
    python pinn_correction_batch.py --predictions-dir predictions/batch_results --vtp-dir vtp_data --epochs 500 --limit 2
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--predictions-dir", type=str, help="Directory with DeepONet CSV predictions"
    )
    input_group.add_argument("--csv-file", type=str, help="Single CSV file to process")

    parser.add_argument("--vtp-dir", type=str, help="Directory with VTP geometry files")
    parser.add_argument("--vtp-file", type=str, help="Single VTP file (required with --csv-file)")

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions/pinn_corrected",
        help="Output directory (default: predictions/pinn_corrected)",
    )

    # PINN parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--n-interior",
        type=int,
        default=DEFAULT_N_INTERIOR,
        help=f"Interior points for physics (default: {DEFAULT_N_INTERIOR})",
    )
    parser.add_argument(
        "--n-wall",
        type=int,
        default=DEFAULT_N_WALL,
        help=f"Wall points (default: {DEFAULT_N_WALL})",
    )
    parser.add_argument(
        "--steady", action="store_true", help="Use steady-state PINN instead of unsteady"
    )

    # Other options
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # Update device
    global DEVICE
    DEVICE = torch.device(args.device)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(output_dir)
    logger.info("=" * 60)
    logger.info("PINN Correction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Mode: {'Steady' if args.steady else 'Unsteady'}")

    # Get list of files to process
    if args.csv_file:
        # Single file mode
        csv_files = [Path(args.csv_file)]
        if not args.vtp_file:
            logger.error("--vtp-file is required when using --csv-file")
            sys.exit(1)
        vtp_files = [Path(args.vtp_file)]
    else:
        # Batch mode
        predictions_dir = Path(args.predictions_dir)
        csv_files = sorted(predictions_dir.glob("*.csv"))
        # Filter out summary files
        csv_files = [
            f
            for f in csv_files
            if "summary" not in f.name.lower() and "failed" not in f.name.lower()
        ]

        if not args.vtp_dir:
            logger.error("--vtp-dir is required when using --predictions-dir")
            sys.exit(1)

        vtp_dir = Path(args.vtp_dir)

        # Match CSV files to VTP files
        vtp_files = []
        matched_csv = []
        for csv_file in csv_files:
            # Remove _corrected suffix if present
            base_name = csv_file.stem.replace("_corrected", "")
            vtp_file = vtp_dir / f"{base_name}.vtp"
            if vtp_file.exists():
                vtp_files.append(vtp_file)
                matched_csv.append(csv_file)
            else:
                logger.warning(f"No matching VTP for {csv_file.name}")

        csv_files = matched_csv

    if not csv_files:
        logger.error("No CSV files found to process")
        sys.exit(1)

    # Apply limit
    if args.limit:
        csv_files = csv_files[: args.limit]
        vtp_files = vtp_files[: args.limit]

    logger.info(f"Processing {len(csv_files)} cases")
    logger.info("=" * 60)

    # Process files
    results = []
    failed = []

    for csv_file, vtp_file in zip(csv_files, vtp_files):
        try:
            result = process_single_case(
                csv_file=csv_file,
                vtp_file=vtp_file,
                output_dir=output_dir,
                epochs=args.epochs,
                n_interior=args.n_interior,
                n_wall=args.n_wall,
                unsteady=not args.steady,
                logger=logger,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {csv_file.name}: {e}")
            import traceback

            traceback.print_exc()
            failed.append({"file": str(csv_file), "error": str(e)})

    # Save summary
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Successful: {len(results)}/{len(csv_files)}")
    logger.info(f"Failed: {len(failed)}")

    if results:
        avg_time = np.mean([r["total_time"] for r in results])
        avg_wss = np.mean([r["wss_mean"] for r in results])
        avg_osi = np.mean([r["osi_mean"] for r in results])
        logger.info(f"Average processing time: {avg_time:.1f}s")
        logger.info(f"Average WSS: {avg_wss:.4f}")
        logger.info(f"Average OSI: {avg_osi:.4f}")

        # Save summary CSV
        import csv

        summary_file = output_dir / "pinn_summary.csv"
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Summary saved to: {summary_file}")

    if failed:
        import csv

        failed_file = output_dir / "pinn_failed.csv"
        with open(failed_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "error"])
            writer.writeheader()
            writer.writerows(failed)
        logger.info(f"Failed files logged to: {failed_file}")

    logger.info("Pipeline completed!")


if __name__ == "__main__":
    main()
