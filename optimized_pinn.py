#!/usr/bin/env python3
# Version 3 source snapshot
"""
Optimized Batch PINN Correction Script
---------------------------------------
Combines optimized PINN training (vectorized time, AMP, early stopping, steady warm start)
with batch processing for multiple VTP/CSV files.

Output structure per case:
  output_dir/
    case_name/
      timesteps/
        flow_t00.csv, flow_t01.csv, ...    (velocity per timestep)
        wss_t00.csv, wss_t01.csv, ...      (WSS vectors per timestep)
        flow_t00.vtp, flow_t01.vtp, ...    (VTP files per timestep)
        time_index.csv                     (timestep mapping)
      hemodynamics_aggregate.csv           (TAWSS, OSI, von Mises per point)
      hemodynamics_aggregate.vtp           (VTP with aggregate fields)
      pinn_model.pt                        (trained PINN model)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Fix OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: pyvista not installed. VTP output will be disabled.")
# Constants
MU = 0.0035  # Dynamic viscosity (Pa.s)
RHO = 1060.0  # Blood density (kg/m³)
NU = MU / RHO  # Kinematic viscosity (m²/s)

DEFAULT_STEADY_EPOCHS = 300
DEFAULT_UNSTEADY_EPOCHS = 500
DEFAULT_NT = 10
DEFAULT_EARLY_STOP_PATIENCE = 100
# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# PINN Model Definition
class PINNCorrection(nn.Module):
    """
    Physics-Informed Neural Network for correcting DeepONet predictions.
    Takes (x, y, z, t) and outputs velocity corrections (du, dv, dw) and pressure p.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))  # outputs: du, dv, dw, p
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Pulsatile Flow Profile
def pulsatile_scaling(t: torch.Tensor, period: float = 1.0) -> torch.Tensor:
    """
    Compute pulsatile flow scaling factor.
    Returns a smooth periodic function representing cardiac cycle variation.
    """
    phase = 2 * np.pi * t / period
    # Approximate cardiac waveform: systolic peak + diastolic baseline
    return 0.6 + 0.4 * torch.sin(phase) + 0.1 * torch.sin(2 * phase)


# Training Functions
def train_steady(
    model: PINNCorrection,
    coords: torch.Tensor,
    baseline_vel: torch.Tensor,
    is_wall: torch.Tensor,
    device: torch.device,
    epochs: int = DEFAULT_STEADY_EPOCHS,
    lr: float = 1e-3,
) -> PINNCorrection:
    """
    Train steady-state correction (t=0) to warm-start the model.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = coords.shape[0]
    t_zero = torch.zeros(N, 1, device=device)
    xyzt = torch.cat([coords, t_zero], dim=1)

    wall_mask = is_wall.bool().squeeze()

    logger.info(f"Steady warm-start training: {epochs} epochs")

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(xyzt)
        du, dv, dw = out[:, 0], out[:, 1], out[:, 2]

        u_corr = baseline_vel[:, 0] + du
        v_corr = baseline_vel[:, 1] + dv
        w_corr = baseline_vel[:, 2] + dw

        # Wall BC: velocity = 0 at walls
        wall_loss = (
            u_corr[wall_mask] ** 2 + v_corr[wall_mask] ** 2 + w_corr[wall_mask] ** 2
        ).mean()

        # Small regularization on corrections
        reg_loss = 0.001 * (du**2 + dv**2 + dw**2).mean()

        loss = wall_loss + reg_loss
        loss.backward()
        optimizer.step()

    logger.info(f"Steady warm-start complete. Final loss: {loss.item():.4e}")
    return model


def train_unsteady_vectorized(
    model: PINNCorrection,
    coords: torch.Tensor,
    baseline_vel: torch.Tensor,
    is_wall: torch.Tensor,
    device: torch.device,
    epochs: int = DEFAULT_UNSTEADY_EPOCHS,
    nt: int = DEFAULT_NT,
    lr: float = 1e-3,
    use_amp: bool = True,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
) -> Tuple[PINNCorrection, List[float]]:
    """
    Train unsteady PINN with vectorized time processing, AMP, and early stopping.

    Key optimizations:
    - Vectorized time: process all timesteps in one forward pass
    - AMP (Automatic Mixed Precision) for faster GPU computation
    - Early stopping to avoid unnecessary epochs
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    N = coords.shape[0]
    wall_mask = is_wall.bool().squeeze()

    # Create time grid
    t_values = torch.linspace(0, 1, nt, device=device)

    # Expand coords for all timesteps: (N * nt, 4)
    coords_expanded = coords.repeat(nt, 1)  # (N*nt, 3)
    t_expanded = t_values.repeat_interleave(N).unsqueeze(1)  # (N*nt, 1)
    xyzt = torch.cat([coords_expanded, t_expanded], dim=1)
    xyzt.requires_grad_(True)

    # Baseline velocity repeated for all timesteps
    baseline_expanded = baseline_vel.repeat(nt, 1)  # (N*nt, 3)

    # Pulsatile scaling for each point-time pair
    pulsatile = pulsatile_scaling(t_expanded.squeeze())  # (N*nt,)

    # Wall mask for all timesteps
    wall_mask_expanded = wall_mask.repeat(nt)

    best_loss = float("inf")
    patience_counter = 0
    losses = []

    logger.info(
        f"Unsteady training: {epochs} epochs, {nt} time steps, AMP={'on' if scaler else 'off'}"
    )

    pbar = tqdm(range(epochs), desc="PINN Training")
    for epoch in pbar:
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                out = model(xyzt)
                du, dv, dw, _p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

                # Corrected velocities with pulsatile modulation
                u_corr = pulsatile * baseline_expanded[:, 0] + du
                v_corr = pulsatile * baseline_expanded[:, 1] + dv
                w_corr = pulsatile * baseline_expanded[:, 2] + dw

                # Wall BC loss
                wall_loss = (
                    u_corr[wall_mask_expanded] ** 2
                    + v_corr[wall_mask_expanded] ** 2
                    + w_corr[wall_mask_expanded] ** 2
                ).mean()

            # Physics loss (computed outside autocast for gradient accuracy)
            # Each output gets its own gradient tensor
            grad_u = torch.autograd.grad(u_corr.sum(), xyzt, create_graph=True, retain_graph=True)[
                0
            ]
            grad_v = torch.autograd.grad(v_corr.sum(), xyzt, create_graph=True, retain_graph=True)[
                0
            ]
            grad_w = torch.autograd.grad(w_corr.sum(), xyzt, create_graph=True, retain_graph=True)[
                0
            ]

            # Extract spatial derivatives: du/dx, dv/dy, dw/dz
            du_dx = grad_u[:, 0]
            dv_dy = grad_v[:, 1]
            dw_dz = grad_w[:, 2]

            # Continuity (incompressibility): div(u) = 0
            continuity = du_dx + dv_dy + dw_dz
            physics_loss = (continuity**2).mean()

            loss = wall_loss + 0.01 * physics_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(xyzt)
            du, dv, dw, _p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

            u_corr = pulsatile * baseline_expanded[:, 0] + du
            v_corr = pulsatile * baseline_expanded[:, 1] + dv
            w_corr = pulsatile * baseline_expanded[:, 2] + dw

            wall_loss = (
                u_corr[wall_mask_expanded] ** 2
                + v_corr[wall_mask_expanded] ** 2
                + w_corr[wall_mask_expanded] ** 2
            ).mean()

            # Each output gets its own gradient tensor
            grad_u = torch.autograd.grad(u_corr.sum(), xyzt, create_graph=True, retain_graph=True)[
                0
            ]
            grad_v = torch.autograd.grad(v_corr.sum(), xyzt, create_graph=True, retain_graph=True)[
                0
            ]
            grad_w = torch.autograd.grad(w_corr.sum(), xyzt, create_graph=True, retain_graph=True)[
                0
            ]

            # Extract spatial derivatives: du/dx, dv/dy, dw/dz
            du_dx = grad_u[:, 0]
            dv_dy = grad_v[:, 1]
            dw_dz = grad_w[:, 2]

            # Continuity (incompressibility): div(u) = 0
            continuity = du_dx + dv_dy + dw_dz
            physics_loss = (continuity**2).mean()

            loss = wall_loss + 0.01 * physics_loss
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        losses.append(loss.item())

        pbar.set_postfix(loss=f"{loss.item():.3e}", wall=f"{wall_loss.item():.3e}")

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    logger.info(f"Training complete. Final loss: {loss.item():.4e}")
    return model, losses


# Hemodynamic Computations
def compute_wss(
    coords: np.ndarray, velocities: np.ndarray, normals: np.ndarray, mu: float = MU
) -> np.ndarray:
    """
    Compute Wall Shear Stress vectors using velocity gradient estimation.

    WSS = mu * (grad(u) . n - (n . grad(u) . n) * n)
    """
    from scipy.spatial import cKDTree

    N = coords.shape[0]
    wss = np.zeros((N, 3))

    tree = cKDTree(coords)
    k = min(20, N)

    for i in range(N):
        _, idx = tree.query(coords[i], k=k)
        neighbors = coords[idx]
        vel_neighbors = velocities[idx]

        # Local coordinate system
        center = neighbors.mean(axis=0)
        local_coords = neighbors - center
        local_vel = vel_neighbors

        # Least squares gradient estimation
        try:
            A = local_coords
            grad_u = np.linalg.lstsq(A, local_vel[:, 0], rcond=None)[0]
            grad_v = np.linalg.lstsq(A, local_vel[:, 1], rcond=None)[0]
            grad_w = np.linalg.lstsq(A, local_vel[:, 2], rcond=None)[0]

            grad_vel = np.array([grad_u, grad_v, grad_w])  # 3x3

            n = normals[i]
            n = n / (np.linalg.norm(n) + 1e-10)

            # Stress vector at wall
            tau = mu * grad_vel @ n

            # Remove normal component to get WSS (tangential)
            tau_n = np.dot(tau, n) * n
            wss[i] = tau - tau_n
        except Exception:
            wss[i] = np.zeros(3)

    return wss


def compute_tawss(wss_over_time: np.ndarray) -> np.ndarray:
    """
    Compute Time-Averaged Wall Shear Stress magnitude.
    wss_over_time: (N, T, 3) array
    Returns: (N,) array of TAWSS magnitudes
    """
    wss_mag = np.linalg.norm(wss_over_time, axis=2)  # (N, T)
    return wss_mag.mean(axis=1)


def compute_osi(wss_over_time: np.ndarray) -> np.ndarray:
    """
    Compute Oscillatory Shear Index.
    OSI = 0.5 * (1 - |mean(WSS)| / mean(|WSS|))
    """
    wss_mag = np.linalg.norm(wss_over_time, axis=2)  # (N, T)
    wss_mean = wss_over_time.mean(axis=1)  # (N, 3)
    wss_mean_mag = np.linalg.norm(wss_mean, axis=1)  # (N,)
    wss_mag_mean = wss_mag.mean(axis=1)  # (N,)

    with np.errstate(divide="ignore", invalid="ignore"):
        osi = 0.5 * (1 - wss_mean_mag / (wss_mag_mean + 1e-10))
    osi = np.clip(osi, 0, 0.5)
    return osi


def compute_von_mises(wss: np.ndarray) -> np.ndarray:
    """
    Compute von Mises stress from WSS vectors.
    For surface stress: sigma_vm = sqrt(tau_x^2 + tau_y^2 + tau_z^2 - tau_x*tau_y - tau_y*tau_z - tau_z*tau_x)
    Simplified for WSS: just use magnitude
    """
    return np.linalg.norm(wss, axis=1)


# Data Loading
def load_deeponet_predictions(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load DeepONet predictions from CSV file."""
    df = pd.read_csv(csv_path)

    # Get coordinates
    coords = df[["x", "y", "z"]].values

    # Get velocity columns
    vel_cols = ["u", "v", "w"]
    if not all(c in df.columns for c in vel_cols):
        vel_cols = ["Velocity:0", "Velocity:1", "Velocity:2"]

    velocities = df[vel_cols].values

    return coords, velocities


def extract_wall_from_vtp(vtp_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract wall points, normals, and identify wall nodes from VTP file.
    Returns: (coords, normals, is_wall mask)
    """
    if not HAS_PYVISTA:
        raise ImportError("pyvista required for VTP processing")

    mesh = pv.read(str(vtp_path))
    coords = np.array(mesh.points)

    # Compute normals
    if mesh.n_cells > 0:
        mesh = mesh.compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True
        )
        if "Normals" in mesh.point_data:
            normals = np.array(mesh.point_data["Normals"])
        else:
            normals = np.zeros_like(coords)
            normals[:, 2] = 1.0
    else:
        normals = np.zeros_like(coords)
        normals[:, 2] = 1.0

    # All points from VTP are wall points
    is_wall = np.ones(len(coords), dtype=bool)

    return coords, normals, is_wall


def match_points(
    deeponet_coords: np.ndarray, vtp_coords: np.ndarray, tolerance: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match DeepONet prediction points to VTP wall surface points.
    Returns: (indices into deeponet_coords, is_wall mask for deeponet_coords)

    DeepONet predictions include interior domain points.
    VTP contains only wall surface points.
    Only points within tolerance distance of wall surface are marked as wall points.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(vtp_coords)
    distances, _ = tree.query(deeponet_coords)

    # Only points close to the wall surface get wall BC
    is_wall = distances < tolerance

    wall_count = is_wall.sum()
    interior_count = len(deeponet_coords) - wall_count
    logger.info(
        f"  Point classification: {wall_count} wall, {interior_count} interior (tolerance={tolerance})"
    )

    return np.arange(len(deeponet_coords)), is_wall


# Output Functions
def save_timestep_csv(
    coords: np.ndarray, velocities: np.ndarray, wss: np.ndarray, timestep: int, output_dir: Path
):
    """Save per-timestep flow and WSS data."""
    timesteps_dir = output_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)

    # Flow CSV
    flow_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "u": velocities[:, 0],
            "v": velocities[:, 1],
            "w": velocities[:, 2],
            "velocity_magnitude": np.linalg.norm(velocities, axis=1),
        }
    )
    flow_df.to_csv(timesteps_dir / f"flow_t{timestep:02d}.csv", index=False)

    # WSS CSV
    wss_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "wss_x": wss[:, 0],
            "wss_y": wss[:, 1],
            "wss_z": wss[:, 2],
            "wss_magnitude": np.linalg.norm(wss, axis=1),
        }
    )
    wss_df.to_csv(timesteps_dir / f"wss_t{timestep:02d}.csv", index=False)


def save_timestep_vtp(
    coords: np.ndarray,
    velocities: np.ndarray,
    wss: np.ndarray,
    timestep: int,
    output_dir: Path,
    original_mesh: Optional[Any] = None,
):
    """Save per-timestep VTP file."""
    if not HAS_PYVISTA:
        return

    timesteps_dir = output_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)

    # Create new point cloud from prediction coordinates
    # (original mesh may have different point count)
    mesh = pv.PolyData(coords)

    mesh.point_data["Velocity"] = velocities
    mesh.point_data["velocity_magnitude"] = np.linalg.norm(velocities, axis=1)
    mesh.point_data["WSS"] = wss
    mesh.point_data["WSS_magnitude"] = np.linalg.norm(wss, axis=1)

    mesh.save(str(timesteps_dir / f"flow_t{timestep:02d}.vtp"))


def save_time_index(nt: int, output_dir: Path, period: float = 1.0):
    """Save time index mapping file."""
    timesteps_dir = output_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)

    t_values = np.linspace(0, period, nt)
    index_df = pd.DataFrame({"timestep": range(nt), "time": t_values, "phase": t_values / period})
    index_df.to_csv(timesteps_dir / "time_index.csv", index=False)


def save_aggregate_hemodynamics(
    coords: np.ndarray,
    tawss: np.ndarray,
    osi: np.ndarray,
    von_mises: np.ndarray,
    output_dir: Path,
    original_mesh: Optional[Any] = None,
):
    """Save aggregate hemodynamic parameters."""
    # CSV
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "TAWSS": tawss,
            "OSI": osi,
            "von_Mises": von_mises,
        }
    )
    df.to_csv(output_dir / "hemodynamics_aggregate.csv", index=False)

    # VTP - create new point cloud from prediction coordinates
    if HAS_PYVISTA:
        mesh = pv.PolyData(coords)

        mesh.point_data["TAWSS"] = tawss
        mesh.point_data["OSI"] = osi
        mesh.point_data["von_Mises"] = von_mises
        mesh.save(str(output_dir / "hemodynamics_aggregate.vtp"))


# Main Processing Function
def process_single_case(
    csv_path: Path,
    vtp_path: Path,
    output_dir: Path,
    device: torch.device,
    steady_epochs: int = DEFAULT_STEADY_EPOCHS,
    unsteady_epochs: int = DEFAULT_UNSTEADY_EPOCHS,
    nt: int = DEFAULT_NT,
    use_amp: bool = True,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
) -> bool:
    """
    Process a single case through the optimized PINN correction pipeline.

    Returns True on success, False on failure.
    """
    case_name = csv_path.stem.replace("_flow", "")
    case_output_dir = output_dir / case_name

    try:
        logger.info(f"Processing: {case_name}")
        start_time = time.time()

        # Load data
        logger.info("  Loading DeepONet predictions...")
        deeponet_coords, deeponet_vel = load_deeponet_predictions(csv_path)

        logger.info("  Loading VTP mesh...")
        vtp_coords, normals, _ = extract_wall_from_vtp(vtp_path)
        original_mesh = pv.read(str(vtp_path)) if HAS_PYVISTA else None

        # Match points
        _, is_wall = match_points(deeponet_coords, vtp_coords)
        logger.info(f"  Matched {is_wall.sum()} wall points out of {len(deeponet_coords)}")

        # Prepare tensors
        coords_t = torch.tensor(deeponet_coords, dtype=torch.float32, device=device)
        vel_t = torch.tensor(deeponet_vel, dtype=torch.float32, device=device)
        is_wall_t = torch.tensor(is_wall, dtype=torch.float32, device=device).unsqueeze(1)

        # Initialize and train PINN
        model = PINNCorrection(hidden_dim=64, num_layers=3).to(device)

        # Steady warm-start
        logger.info("  Training steady warm-start...")
        model = train_steady(model, coords_t, vel_t, is_wall_t, device, epochs=steady_epochs)

        # Unsteady training
        logger.info("  Training unsteady PINN...")
        model, losses = train_unsteady_vectorized(
            model,
            coords_t,
            vel_t,
            is_wall_t,
            device,
            epochs=unsteady_epochs,
            nt=nt,
            use_amp=use_amp,
            early_stop_patience=early_stop_patience,
        )

        # Save model
        case_output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), case_output_dir / "pinn_model.pt")

        # Generate corrected velocities for all timesteps
        logger.info("  Computing corrected velocities and hemodynamics...")
        model.eval()

        t_values = torch.linspace(0, 1, nt, device=device)
        wss_over_time = np.zeros((len(deeponet_coords), nt, 3))

        with torch.no_grad():
            for ti, t in enumerate(t_values):
                t_tensor = torch.full((len(deeponet_coords), 1), t.item(), device=device)
                xyzt = torch.cat([coords_t, t_tensor], dim=1)

                out = model(xyzt)
                du, dv, dw = out[:, 0], out[:, 1], out[:, 2]

                pulsatile = pulsatile_scaling(t_tensor.squeeze())

                u_corr = pulsatile * vel_t[:, 0] + du
                v_corr = pulsatile * vel_t[:, 1] + dv
                w_corr = pulsatile * vel_t[:, 2] + dw

                velocities_t = torch.stack([u_corr, v_corr, w_corr], dim=1).cpu().numpy()

                # Compute WSS for this timestep
                wss_t = compute_wss(deeponet_coords, velocities_t, normals)
                wss_over_time[:, ti, :] = wss_t

                # Save timestep data
                save_timestep_csv(deeponet_coords, velocities_t, wss_t, ti, case_output_dir)
                save_timestep_vtp(
                    deeponet_coords, velocities_t, wss_t, ti, case_output_dir, original_mesh
                )

        # Save time index
        save_time_index(nt, case_output_dir)

        # Compute and save aggregate hemodynamics
        tawss = compute_tawss(wss_over_time)
        osi = compute_osi(wss_over_time)

        # Use time-averaged WSS for von Mises (or last timestep)
        von_mises = compute_von_mises(wss_over_time.mean(axis=1))

        save_aggregate_hemodynamics(
            deeponet_coords, tawss, osi, von_mises, case_output_dir, original_mesh
        )

        elapsed = time.time() - start_time
        logger.info(f"  Completed in {elapsed:.1f}s")

        return True

    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def find_matching_pairs(predictions_dir: Path, vtp_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find matching CSV and VTP file pairs.
    """
    pairs = []

    # Get all CSV files
    csv_files = list(predictions_dir.glob("*.csv"))

    for csv_file in csv_files:
        # Extract case name (remove _flow suffix if present)
        case_name = csv_file.stem.replace("_flow", "")

        # Try to find matching VTP
        vtp_candidates = [
            vtp_dir / f"{case_name}.vtp",
            vtp_dir / f"{case_name}_cut1.vtp",
        ]

        for vtp_path in vtp_candidates:
            if vtp_path.exists():
                pairs.append((csv_file, vtp_path))
                break

    return pairs


# CLI Entry Point
def main():
    parser = argparse.ArgumentParser(
        description="Optimized Batch PINN Correction for hemodynamic predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all cases in batch
  python optimized_pinn.py --predictions-dir predictions/batch_results --vtp-dir vtp_data

  # Process single case
  python optimized_pinn.py --csv-file path/to/predictions.csv --vtp-file path/to/mesh.vtp

  # Custom settings
  python optimized_pinn.py --predictions-dir preds --vtp-dir vtps --steady-epochs 500 --unsteady-epochs 1000 --nt 20
        """,
    )

    # Batch mode arguments
    parser.add_argument(
        "--predictions-dir", type=Path, help="Directory containing DeepONet prediction CSVs"
    )
    parser.add_argument("--vtp-dir", type=Path, help="Directory containing VTP mesh files")

    # Single file mode arguments
    parser.add_argument("--csv-file", type=Path, help="Single CSV file to process")
    parser.add_argument("--vtp-file", type=Path, help="Single VTP file to use")

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions/optimized_pinn"),
        help="Output directory (default: predictions/optimized_pinn)",
    )

    # Training parameters
    parser.add_argument(
        "--steady-epochs",
        type=int,
        default=DEFAULT_STEADY_EPOCHS,
        help=f"Epochs for steady warm-start (default: {DEFAULT_STEADY_EPOCHS})",
    )
    parser.add_argument(
        "--unsteady-epochs",
        type=int,
        default=DEFAULT_UNSTEADY_EPOCHS,
        help=f"Epochs for unsteady training (default: {DEFAULT_UNSTEADY_EPOCHS})",
    )
    parser.add_argument(
        "--nt", type=int, default=DEFAULT_NT, help=f"Number of time steps (default: {DEFAULT_NT})"
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help=f"Early stopping patience (default: {DEFAULT_EARLY_STOP_PATIENCE})",
    )

    # Options
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--limit", type=int, help="Limit number of cases to process (for testing)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Validate arguments
    single_mode = args.csv_file is not None and args.vtp_file is not None
    batch_mode = args.predictions_dir is not None and args.vtp_dir is not None

    if not single_mode and not batch_mode:
        parser.error(
            "Must specify either (--csv-file and --vtp-file) or (--predictions-dir and --vtp-dir)"
        )

    device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    logger.info("=" * 60)
    logger.info("Optimized PINN Correction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"AMP: {'enabled' if use_amp else 'disabled'}")
    logger.info(f"Steady epochs: {args.steady_epochs}")
    logger.info(f"Unsteady epochs: {args.unsteady_epochs}")
    logger.info(f"Time steps: {args.nt}")
    logger.info(f"Early stop patience: {args.early_stop_patience}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if single_mode:
        # Single file mode
        logger.info(f"Processing single case: {args.csv_file.name}")
        success = process_single_case(
            args.csv_file,
            args.vtp_file,
            args.output_dir,
            device,
            steady_epochs=args.steady_epochs,
            unsteady_epochs=args.unsteady_epochs,
            nt=args.nt,
            use_amp=use_amp,
            early_stop_patience=args.early_stop_patience,
        )
        sys.exit(0 if success else 1)

    else:
        # Batch mode
        pairs = find_matching_pairs(args.predictions_dir, args.vtp_dir)

        if args.limit:
            pairs = pairs[: args.limit]

        logger.info(f"Found {len(pairs)} matching CSV/VTP pairs")
        logger.info("=" * 60)

        successful = 0
        failed = 0

        for csv_path, vtp_path in pairs:
            success = process_single_case(
                csv_path,
                vtp_path,
                args.output_dir,
                device,
                steady_epochs=args.steady_epochs,
                unsteady_epochs=args.unsteady_epochs,
                nt=args.nt,
                use_amp=use_amp,
                early_stop_patience=args.early_stop_patience,
            )

            if success:
                successful += 1
            else:
                failed += 1

        logger.info("=" * 60)
        logger.info(f"Completed: {successful} successful, {failed} failed")
        logger.info("=" * 60)

        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
