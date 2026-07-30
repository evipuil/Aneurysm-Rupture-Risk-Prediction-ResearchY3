# Version 12 source snapshot
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration

VTP_DIR = os.environ.get("v12_VTP_DIR", "vtp_data")
OUTPUT_DIR = os.environ.get("v12_OUTPUT_DIR", "predictions/full_accuracy")
TRAINING_LOG_SUBDIR = "training_logs"

STEADY_EPOCHS = int(os.environ.get("v12_STEADY_EPOCHS", 1000))
UNSTEADY_EPOCHS = int(os.environ.get("v12_UNSTEADY_EPOCHS", 2000))
LBFGS_ITERS = int(os.environ.get("v12_LBFGS_ITERS", 50))
NT = int(os.environ.get("v12_NT", 10))
N_INTERIOR = int(os.environ.get("v12_N_INTERIOR", 4096))
N_WALL = int(os.environ.get("v12_N_WALL", 2048))
N_INLET = int(os.environ.get("v12_N_INLET", 2048))
N_OUTLET = int(os.environ.get("v12_N_OUTLET", N_INLET))
COLLOC_BATCH = int(os.environ.get("v12_COLLOC_BATCH", 2048))
T_END = 1.0
MU = 0.0035
RHO = 1060.0
NU = MU / RHO

FLOW_RATE = float(os.environ.get("v12_FLOW_RATE", 0.2))
HIDDEN_DIM = int(os.environ.get("v12_HIDDEN_DIM", 128))
N_LAYERS = int(os.environ.get("v12_N_LAYERS", 5))
FOURIER_MODES = int(os.environ.get("v12_FOURIER_MODES", 16))
FOURIER_SIGMA = float(os.environ.get("v12_FOURIER_SIGMA", 2.0))

LR = float(os.environ.get("v12_LR", 1e-3))
WALL_LOSS_WEIGHT = float(os.environ.get("v12_WALL_W", 5.0))
INLET_LOSS_WEIGHT = float(os.environ.get("v12_INLET_W", 5.0))
OUTLET_LOSS_WEIGHT = float(os.environ.get("v12_OUTLET_W", 2.0))
PHYSICS_LOSS_WEIGHT = float(os.environ.get("v12_PHYS_W", 1.0))
PERIODIC_LOSS_WEIGHT = float(os.environ.get("v12_PERIODIC_W", 1.0))
GRAD_CLIP = float(os.environ.get("v12_GRAD_CLIP", 1.0))
LOG_EVERY = int(os.environ.get("v12_LOG_EVERY", 25))
LOSS_REF_EMA = float(os.environ.get("v12_LOSS_REF_EMA", 0.98))
SCHEDULER_KIND = os.environ.get("v12_SCHEDULER", "warm_restarts").strip().lower()
SCHEDULER_ETA_MIN = float(os.environ.get("v12_ETA_MIN", 1e-6))
SCHEDULER_RESTART_T0 = int(os.environ.get("v12_RESTART_T0", max(50, UNSTEADY_EPOCHS // 5)))
EARLY_STOP_PATIENCE = int(os.environ.get("v12_EARLY_STOP_PATIENCE", 200))
SAVE_CHECKPOINT_EVERY = int(os.environ.get("v12_SAVE_CHECKPOINT_EVERY", 0))
FIXED_COLLOCATION = os.environ.get("v12_FIXED_COLLOCATION", "0").lower() in {"1", "true", "yes"}

LIMIT = int(os.environ["v12_LIMIT"]) if os.environ.get("v12_LIMIT") else None
SEED = 42
UNSTEADY = os.environ.get("v12_UNSTEADY", "1").lower() in {"1", "true", "yes"}
REQUIRE_GPU = os.environ.get("v12_REQUIRE_GPU", "1").lower() in {"1", "true", "yes"}
CUDA_INDEX = int(os.environ.get("v12_CUDA_DEVICE", 0))


def _resolve_device():
    if not torch.cuda.is_available():
        if REQUIRE_GPU:
            raise RuntimeError("CUDA required. Set v12_REQUIRE_GPU=0 to allow CPU.")
        return torch.device("cpu")
    idx = max(0, min(CUDA_INDEX, torch.cuda.device_count() - 1))
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


DEVICE = _resolve_device()
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

np.random.seed(SEED)
torch.manual_seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
TRAINING_LOG_DIR = Path(OUTPUT_DIR) / TRAINING_LOG_SUBDIR
TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("v12PINNPipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)
_fh = logging.FileHandler(Path(OUTPUT_DIR) / "pinn_pipeline.log")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# Geometry helpers


def _gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def compute_case_scales(surface, inlet_cap):
    bounds = surface.bounds
    length_scale = float(
        max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-6)
    )
    inlet_area = (
        float(inlet_cap.area)
        if hasattr(inlet_cap, "area") and inlet_cap.area > 0
        else float(max(inlet_cap.n_points, 1))
    )
    inlet_speed = FLOW_RATE / max(inlet_area, 1e-8)
    velocity_scale = float(max(inlet_speed, 1e-6))
    time_scale = float(max(length_scale / velocity_scale, 1e-6))
    pressure_scale = float(max(RHO * velocity_scale * velocity_scale, 1e-6))
    reynolds = float(max(RHO * velocity_scale * length_scale / MU, 1.0))
    return length_scale, velocity_scale, time_scale, pressure_scale, reynolds


def build_model_input(x, t, length_scale: float, time_scale: float):
    return torch.cat([x / length_scale, t / time_scale], dim=1)


def _subsample_points(points: np.ndarray, limit: int):
    if len(points) <= limit:
        return points.astype(np.float32)
    idx = np.random.choice(len(points), limit, replace=False)
    return points[idx].astype(np.float32)


def _caps_to_points(caps):
    if not caps:
        return np.empty((0, 3), dtype=np.float32)
    pieces = [cap.points.astype(np.float32) for cap in caps if cap.n_points > 0]
    if not pieces:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(pieces).astype(np.float32)


def compute_boundary_normal(points, k=10):
    if len(points) < 3:
        centre = points.mean(axis=0) if len(points) else np.zeros(3)
        return centre, np.array([0.0, 0.0, 1.0])
    k = min(k, len(points) - 1)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    indices = np.argsort(distances, axis=1)[:, : k + 1]
    normals = []
    for idx_row in indices:
        neigh = np.take(points, np.asarray(idx_row, dtype=int), axis=0)
        centered = neigh - neigh.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        _, eigenvectors = np.linalg.eigh(cov)
        n = eigenvectors[:, 0]
        normals.append(n / (np.linalg.norm(n) + 1e-12))
    return points.mean(axis=0), np.mean(normals, axis=0)


def _extract_boundary_loops(surface):
    edges = surface.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    )
    if edges.n_points == 0:
        return []
    conn = edges.connectivity()
    labels = np.unique(conn["RegionId"])
    return [conn.threshold([lb - 0.1, lb + 0.1]) for lb in labels]


def _cap_loop(loop):
    if loop.n_points < 3:
        return loop
    try:
        return pv.PolyData(loop.points).delaunay_2d()
    except Exception:
        return loop


def pick_inlet(surface):
    """Return inlet cap, per-point velocity target, and the remaining boundary caps."""
    loops = _extract_boundary_loops(surface)
    if not loops:
        edges = surface.extract_feature_edges(boundary_edges=True)
        inlet_cap = pv.PolyData(edges.points if edges.n_points > 0 else surface.points[:100])
        outlet_caps = []
    else:
        caps = [_cap_loop(lp) for lp in loops]
        inlet_cap = max(
            caps, key=lambda c: c.area if hasattr(c, "area") and c.area > 0 else c.n_points
        )
        outlet_caps = [cap for cap in caps if cap is not inlet_cap]

    centre, normal = compute_boundary_normal(inlet_cap.points, k=10)
    vol_center = surface.points.mean(axis=0)
    if np.dot(normal, vol_center - centre) < 0:
        normal = -normal
    inlet_area = (
        float(inlet_cap.area)
        if hasattr(inlet_cap, "area") and inlet_cap.area > 0
        else float(max(inlet_cap.n_points, 1))
    )
    inlet_speed = FLOW_RATE / max(inlet_area, 1e-8)
    return inlet_cap, (inlet_speed * normal).astype(np.float32), outlet_caps


def generate_interior_points(surface, n_points: int):
    logger.info("Generating %d interior points", n_points)
    try:
        filled = surface.fill_holes(hole_size=1000.0)
    except Exception:
        filled = surface

    bounds = filled.bounds
    n_candidates = n_points * 20
    xs = np.random.uniform(bounds[0], bounds[1], n_candidates)
    ys = np.random.uniform(bounds[2], bounds[3], n_candidates)
    zs = np.random.uniform(bounds[4], bounds[5], n_candidates)
    candidates = np.column_stack([xs, ys, zs])

    interior = None
    try:
        sel = pv.PolyData(candidates).select_enclosed_points(
            filled, tolerance=0.0, check_surface=False
        )
        mask = sel["SelectedPoints"].astype(bool)
        interior = candidates[mask]
    except Exception:
        pass

    if interior is None or len(interior) < n_points // 10:
        tree = KDTree(filled.points)
        dists, _ = tree.query(candidates, k=1)
        interior = candidates[dists < np.percentile(dists, 50)]

    if len(interior) < n_points:
        n_extra = min(n_points - len(interior), len(filled.points))
        extra_idx = np.random.choice(len(filled.points), n_extra, replace=False)
        interior = (
            np.vstack([interior, filled.points[extra_idx]])
            if len(interior)
            else filled.points[extra_idx]
        )
    else:
        idx = np.random.choice(len(interior), n_points, replace=False)
        interior = interior[idx]
    return interior.astype(np.float32)


def extract_wall_points(surface, n_wall: int):
    if surface.point_normals is None:
        surface = surface.compute_normals(point_normals=True, cell_normals=False)
    pts = surface.points.astype(np.float32)
    normals = surface.point_normals.astype(np.float32)
    if len(pts) > n_wall:
        idx = np.random.choice(len(pts), n_wall, replace=False)
        pts = pts[idx]
        normals = normals[idx]
    return pts, normals


# Model
class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, modes: int, sigma: float):
        super().__init__()
        B = torch.randn(in_dim, modes) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj), x], dim=-1)


class PINNNet(nn.Module):
    def __init__(
        self, in_dim=4, hidden=HIDDEN_DIM, layers=N_LAYERS, modes=FOURIER_MODES, sigma=FOURIER_SIGMA
    ):
        super().__init__()
        self.features = FourierFeatures(in_dim, modes, sigma)
        feat_dim = 2 * modes + in_dim
        self.input_proj = nn.Linear(feat_dim, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(
                nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            )
        self.head = nn.Linear(hidden, 4)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = F.silu(self.input_proj(self.features(x)))
        for block in self.blocks:
            h = F.silu(h + block(h))
        return self.head(h)


def pulsatile_scale(t, t_end: float = T_END):
    return 1.0 + 0.6 * torch.sin(2 * np.pi * t / t_end)


# Physics loss
def physics_residual(model, xt_int, unsteady: bool, reynolds: float):
    """Compute Navier-Stokes residuals at collocation points."""
    xt_int = xt_int.requires_grad_(True)
    field = model(xt_int)
    p = field[:, 0:1]
    u = field[:, 1:2]
    v = field[:, 2:3]
    w = field[:, 3:4]

    u_g = _gradients(u, xt_int)
    v_g = _gradients(v, xt_int)
    w_g = _gradients(w, xt_int)
    p_g = _gradients(p, xt_int)

    u_x, u_y, u_z, u_t = u_g.split(1, dim=1)
    v_x, v_y, v_z, v_t = v_g.split(1, dim=1)
    w_x, w_y, w_z, w_t = w_g.split(1, dim=1)
    p_x, p_y, p_z, _ = p_g.split(1, dim=1)

    u_xx = _gradients(u_x, xt_int)[:, 0:1]
    u_yy = _gradients(u_y, xt_int)[:, 1:2]
    u_zz = _gradients(u_z, xt_int)[:, 2:3]
    v_xx = _gradients(v_x, xt_int)[:, 0:1]
    v_yy = _gradients(v_y, xt_int)[:, 1:2]
    v_zz = _gradients(v_z, xt_int)[:, 2:3]
    w_xx = _gradients(w_x, xt_int)[:, 0:1]
    w_yy = _gradients(w_y, xt_int)[:, 1:2]
    w_zz = _gradients(w_z, xt_int)[:, 2:3]

    adv_u = u * u_x + v * u_y + w * u_z
    adv_v = u * v_x + v * v_y + w * v_z
    adv_w = u * w_x + v * w_y + w * w_z
    lap_u = u_xx + u_yy + u_zz
    lap_v = v_xx + v_yy + v_zz
    lap_w = w_xx + w_yy + w_zz

    inv_re = 1.0 / max(reynolds, 1.0)
    mom_u = (u_t if unsteady else 0.0) + adv_u + p_x - inv_re * lap_u
    mom_v = (v_t if unsteady else 0.0) + adv_v + p_y - inv_re * lap_v
    mom_w = (w_t if unsteady else 0.0) + adv_w + p_z - inv_re * lap_w
    cont = u_x + v_y + w_z

    return mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()


def sample_collocation(x_pool, unsteady, batch_size, device):
    """Return (batch_size, 4) -- (x, y, z, t) sampled uniformly from the interior."""
    n = x_pool.shape[0]
    idx = torch.randint(0, n, (batch_size,), device=device)
    x = x_pool[idx]
    if unsteady:
        t = torch.rand(batch_size, 1, device=device) * T_END
    else:
        t = torch.zeros(batch_size, 1, device=device)
    return torch.cat([x, t], dim=1)


def train_pinn(
    x_int,
    x_inlet,
    x_outlet,
    x_wall,
    inlet_target,
    unsteady: bool,
    epochs: int,
    case_name: str,
    length_scale: float,
    velocity_scale: float,
    time_scale: float,
    pressure_scale: float,
    reynolds: float,
):
    """Train PINN with adaptive loss weights + optional L-BFGS polish (steady)."""
    model = PINNNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if SCHEDULER_KIND == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=SCHEDULER_ETA_MIN,
        )
    elif SCHEDULER_KIND == "none":
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(10, min(epochs, SCHEDULER_RESTART_T0)),
            T_mult=1,
            eta_min=SCHEDULER_ETA_MIN,
        )

    x_int = x_int.to(DEVICE)
    x_inlet = x_inlet.to(DEVICE)
    x_outlet = x_outlet.to(DEVICE)
    x_wall = x_wall.to(DEVICE)
    inlet_target_t = (
        torch.tensor(inlet_target, dtype=torch.float32, device=DEVICE).view(1, 3) / velocity_scale
    )

    phys_ref: float = 0.0
    wall_ref: float = 0.0
    inlet_ref: float = 0.0
    outlet_ref: float = 0.0
    periodic_ref: float = 0.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    pbar = tqdm(range(epochs), desc=f"PINN {case_name} ({'unsteady' if unsteady else 'steady'})")

    xt_int_fixed = None
    if FIXED_COLLOCATION:
        try:
            xt_int_fixed = sample_collocation(x_int, unsteady, COLLOC_BATCH, DEVICE)
            logger.info("Using fixed collocation batch for training (reduces sampling variance)")
        except Exception:
            xt_int_fixed = None

    for epoch in pbar:
        t_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        if xt_int_fixed is not None:
            xt_int = xt_int_fixed
        else:
            xt_int = sample_collocation(x_int, unsteady, COLLOC_BATCH, DEVICE)
        xt_int = build_model_input(xt_int[:, :3], xt_int[:, 3:4], length_scale, time_scale)
        loss_phys = physics_residual(model, xt_int, unsteady, reynolds)

        if unsteady:
            t_inlet = torch.rand(x_inlet.shape[0], 1, device=DEVICE) * T_END
            scale = pulsatile_scale(t_inlet)
            target = scale * inlet_target_t
        else:
            t_inlet = torch.zeros(x_inlet.shape[0], 1, device=DEVICE)
            target = inlet_target_t
        xt_inlet = build_model_input(x_inlet, t_inlet, length_scale, time_scale)
        loss_inlet = (model(xt_inlet)[:, 1:4] - target).pow(2).mean()

        if unsteady:
            t_wall = torch.rand(x_wall.shape[0], 1, device=DEVICE) * T_END
        else:
            t_wall = torch.zeros(x_wall.shape[0], 1, device=DEVICE)
        xt_wall = build_model_input(x_wall, t_wall, length_scale, time_scale)
        loss_wall = model(xt_wall)[:, 1:4].pow(2).mean()

        if x_outlet.numel() > 0:
            if unsteady:
                t_outlet = torch.rand(x_outlet.shape[0], 1, device=DEVICE) * T_END
            else:
                t_outlet = torch.zeros(x_outlet.shape[0], 1, device=DEVICE)
            xt_outlet = build_model_input(x_outlet, t_outlet, length_scale, time_scale)
            loss_outlet = model(xt_outlet)[:, 0:1].pow(2).mean()
        else:
            loss_outlet = model(xt_int)[:, 0:1].mean().pow(2)

        if unsteady:
            period_count = min(512, x_int.shape[0])
            period_idx = torch.randperm(x_int.shape[0], device=DEVICE)[:period_count]
            x_periodic = x_int[period_idx]
            t_period_start = torch.zeros(period_count, 1, device=DEVICE)
            t_end = torch.full((period_count, 1), T_END, device=DEVICE)
            xt_start = build_model_input(x_periodic, t_period_start, length_scale, time_scale)
            xt_end = build_model_input(x_periodic, t_end, length_scale, time_scale)
            loss_periodic = (model(xt_start) - model(xt_end)).pow(2).mean()
        else:
            loss_periodic = torch.zeros(1, device=DEVICE)

        if phys_ref == 0.0:
            phys_ref = float(loss_phys.detach()) + 1e-12
            wall_ref = float(loss_wall.detach()) + 1e-12
            inlet_ref = float(loss_inlet.detach()) + 1e-12
            outlet_ref = float(loss_outlet.detach()) + 1e-12
            periodic_ref = float(loss_periodic.detach()) + 1e-12
        else:
            m = min(max(LOSS_REF_EMA, 0.0), 0.9999)
            phys_ref = m * phys_ref + (1.0 - m) * (float(loss_phys.detach()) + 1e-12)
            wall_ref = m * wall_ref + (1.0 - m) * (float(loss_wall.detach()) + 1e-12)
            inlet_ref = m * inlet_ref + (1.0 - m) * (float(loss_inlet.detach()) + 1e-12)
            outlet_ref = m * outlet_ref + (1.0 - m) * (float(loss_outlet.detach()) + 1e-12)
            periodic_ref = m * periodic_ref + (1.0 - m) * (float(loss_periodic.detach()) + 1e-12)
        loss = (
            PHYSICS_LOSS_WEIGHT * loss_phys / phys_ref
            + WALL_LOSS_WEIGHT * loss_wall / wall_ref
            + INLET_LOSS_WEIGHT * loss_inlet / inlet_ref
            + OUTLET_LOSS_WEIGHT * loss_outlet / outlet_ref
            + (PERIODIC_LOSS_WEIGHT * loss_periodic / periodic_ref if unsteady else 0.0)
        )
        loss.backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        lv = float(loss.item())
        if lv < best_loss:
            best_loss = lv
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            try:
                TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
                best_path = Path(TRAINING_LOG_DIR) / f"{case_name}_best.pt"
                torch.save(model.state_dict(), best_path)
                logger.info("Saved best checkpoint: %s (epoch %d)", best_path, epoch + 1)
            except Exception as exc:
                logger.warning("Failed to save best checkpoint: %s", exc)
            no_improve = 0
        else:
            no_improve += 1

        if SAVE_CHECKPOINT_EVERY > 0 and ((epoch + 1) % SAVE_CHECKPOINT_EVERY == 0):
            try:
                ckpt_path = Path(TRAINING_LOG_DIR) / f"{case_name}_ckpt_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                logger.info("Saved periodic checkpoint: %s", ckpt_path)
            except Exception as exc:
                logger.warning("Failed to save periodic checkpoint: %s", exc)

        if EARLY_STOP_PATIENCE > 0 and no_improve >= EARLY_STOP_PATIENCE:
            logger.info(
                "Early stopping triggered (no improvement for %d epochs)", EARLY_STOP_PATIENCE
            )
            break
        history.append(
            {
                "epoch": epoch + 1,
                "total_loss": lv,
                "physics_loss": float(loss_phys.item()),
                "wall_loss": float(loss_wall.item()),
                "inlet_loss": float(loss_inlet.item()),
                "outlet_loss": float(loss_outlet.item()),
                "periodic_loss": float(loss_periodic.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": float(time.time() - t_start),
            }
        )

        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                "[%s][%s] epoch %d/%d | total=%.3e phys=%.3e wall=%.3e inlet=%.3e",
                case_name,
                "unsteady" if unsteady else "steady",
                epoch + 1,
                epochs,
                lv,
                loss_phys.item(),
                loss_wall.item(),
                loss_inlet.item(),
            )
            pbar.set_postfix({"total": f"{lv:.3e}"})

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    if not unsteady and LBFGS_ITERS > 0:
        logger.info("[%s] L-BFGS polish for %d iterations", case_name, LBFGS_ITERS)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=LBFGS_ITERS,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            xt_int_l = sample_collocation(
                x_int, unsteady=False, batch_size=x_int.shape[0], device=DEVICE
            )
            xt_int_l = build_model_input(
                xt_int_l[:, :3], xt_int_l[:, 3:4], length_scale, time_scale
            )
            loss_p = physics_residual(model, xt_int_l, unsteady=False, reynolds=reynolds)
            t_z_inlet = torch.zeros(x_inlet.shape[0], 1, device=DEVICE)
            loss_i = (
                (
                    model(build_model_input(x_inlet, t_z_inlet, length_scale, time_scale))[:, 1:4]
                    - inlet_target_t
                )
                .pow(2)
                .mean()
            )
            t_z_wall = torch.zeros(x_wall.shape[0], 1, device=DEVICE)
            loss_w = (
                model(build_model_input(x_wall, t_z_wall, length_scale, time_scale))[:, 1:4]
                .pow(2)
                .mean()
            )
            if x_outlet.numel() > 0:
                t_z_outlet = torch.zeros(x_outlet.shape[0], 1, device=DEVICE)
                loss_o = (
                    model(build_model_input(x_outlet, t_z_outlet, length_scale, time_scale))[:, 0:1]
                    .pow(2)
                    .mean()
                )
            else:
                loss_o = torch.zeros(1, device=DEVICE)
            loss_l = (
                PHYSICS_LOSS_WEIGHT * loss_p / phys_ref
                + WALL_LOSS_WEIGHT * loss_w / wall_ref
                + INLET_LOSS_WEIGHT * loss_i / inlet_ref
                + OUTLET_LOSS_WEIGHT * loss_o / outlet_ref
            )
            loss_l.backward()
            return loss_l

        try:
            lbfgs.step(closure)
        except RuntimeError as exc:
            logger.warning("L-BFGS skipped: %s", exc)

    stats = {
        "mode": "unsteady" if unsteady else "steady",
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_total_loss": float(best_loss),
        "final_physics_loss": float(history[-1]["physics_loss"] if history else 0.0),
        "final_wall_loss": float(history[-1]["wall_loss"] if history else 0.0),
        "final_inlet_loss": float(history[-1]["inlet_loss"] if history else 0.0),
        "avg_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history]))
        if history
        else 0.0,
    }
    return model, history, stats


# Post-training evaluation + hemodynamic metrics
def evaluate_flow(
    model,
    x_all,
    unsteady: bool,
    length_scale: float,
    velocity_scale: float,
    time_scale: float,
    pressure_scale: float,
):
    """Mean-in-time velocity/pressure over NT samples for unsteady, else single call."""
    model.eval()
    with torch.no_grad():
        if unsteady:
            vel_sum = torch.zeros(x_all.shape[0], 3, device=DEVICE)
            pres_sum = torch.zeros(x_all.shape[0], 1, device=DEVICE)
            ts = torch.linspace(0, T_END, NT, device=DEVICE)
            for t_val in ts:
                t_col = torch.full((x_all.shape[0], 1), float(t_val), device=DEVICE)
                field = model(build_model_input(x_all, t_col, length_scale, time_scale))
                pres_sum += field[:, 0:1] * pressure_scale
                vel_sum += field[:, 1:4] * velocity_scale
            return (pres_sum / len(ts)).cpu().numpy(), (vel_sum / len(ts)).cpu().numpy()
        t_col = torch.zeros(x_all.shape[0], 1, device=DEVICE)
        field = model(build_model_input(x_all, t_col, length_scale, time_scale))
        return (field[:, 0:1] * pressure_scale).cpu().numpy(), (
            field[:, 1:4] * velocity_scale
        ).cpu().numpy()


def compute_wss(
    field_hat, xt_wall, wall_normals, length_scale: float, velocity_scale: float, mu=MU
):
    u = field_hat[:, 1:2]
    v = field_hat[:, 2:3]
    w = field_hat[:, 3:4]
    u_g = _gradients(u, xt_wall)
    v_g = _gradients(v, xt_wall)
    w_g = _gradients(w, xt_wall)

    grad = torch.stack([u_g[:, :3], v_g[:, :3], w_g[:, :3]], dim=1)
    grad = (velocity_scale / max(length_scale, 1e-6)) * grad

    normals = wall_normals / (torch.norm(wall_normals, dim=1, keepdim=True) + 1e-12)
    strain = grad + grad.transpose(1, 2)
    traction = mu * torch.bmm(strain, normals.unsqueeze(-1)).squeeze(-1)
    normal_component = (traction * normals).sum(dim=1, keepdim=True)
    return traction - normal_component * normals


def wss_history_unsteady(
    model,
    x_wall,
    wall_normals,
    nt=NT,
    length_scale: float = 1.0,
    velocity_scale: float = 1.0,
    time_scale: float = 1.0,
):
    history = []
    ts = np.linspace(0, T_END, nt)
    for t in ts:
        t_col = torch.full((x_wall.shape[0], 1), float(t), device=DEVICE)
        with torch.enable_grad():
            xt_wall = build_model_input(x_wall, t_col, length_scale, time_scale).requires_grad_(
                True
            )
            field = model(xt_wall)
            wss = compute_wss(field, xt_wall, wall_normals, length_scale, velocity_scale)
        history.append(wss.detach())
    return history, ts


def compute_osi(wss_history):
    stack = torch.stack(wss_history, dim=0)
    mean_vec = stack.mean(dim=0)
    mean_mag = torch.norm(stack, dim=-1).mean(dim=0)
    vec_mag = torch.norm(mean_vec, dim=-1)
    return 0.5 * (1.0 - vec_mag / (mean_mag + 1e-8))


def compute_tawss(wss_history):
    mags = torch.stack([torch.norm(w, dim=-1) for w in wss_history], dim=0)
    return mags.mean(dim=0)


# Per-case processing
def _write_training_logs(case_name, history, summary):
    if history:
        with open(TRAINING_LOG_DIR / f"{case_name}_epoch_losses.csv", "w", newline="") as f:
            fields = list(history[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(history)
    summary_path = TRAINING_LOG_DIR / "case_training_summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)


def process_case(vtp_file: Path, output_dir: Path, unsteady: bool):
    case_name = vtp_file.stem
    start_time = time.time()
    logger.info("Processing %s", case_name)

    mesh = pv.read(vtp_file)
    surface = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh

    xyz = generate_interior_points(surface, N_INTERIOR)
    wall_pts, wall_normals = extract_wall_points(surface, N_WALL)
    inlet_cap, inlet_target, outlet_caps = pick_inlet(surface)
    length_scale, velocity_scale, time_scale, pressure_scale, reynolds = compute_case_scales(
        surface, inlet_cap
    )

    inlet_pts = inlet_cap.points.astype(np.float32)
    if len(inlet_pts) > N_INLET:
        inlet_pts = _subsample_points(inlet_pts, N_INLET)

    outlet_pts = _caps_to_points(outlet_caps)
    if len(outlet_pts) > N_OUTLET:
        outlet_pts = _subsample_points(outlet_pts, N_OUTLET)

    x_int = torch.tensor(xyz, dtype=torch.float32, device=DEVICE)
    x_wall = torch.tensor(wall_pts, dtype=torch.float32, device=DEVICE)
    n_wall_t = torch.tensor(wall_normals, dtype=torch.float32, device=DEVICE)
    x_inlet = torch.tensor(inlet_pts, dtype=torch.float32, device=DEVICE)
    x_outlet = torch.tensor(outlet_pts, dtype=torch.float32, device=DEVICE)

    train_start = time.time()
    epochs = UNSTEADY_EPOCHS if unsteady else STEADY_EPOCHS
    model, history, train_stats = train_pinn(
        x_int,
        x_inlet,
        x_outlet,
        x_wall,
        inlet_target,
        unsteady,
        epochs,
        case_name,
        length_scale,
        velocity_scale,
        time_scale,
        pressure_scale,
        reynolds,
    )
    train_time = time.time() - train_start

    case_dir = Path(output_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    timesteps_dir = case_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)

    x_all = torch.tensor(xyz, dtype=torch.float32, device=DEVICE)
    pressure_mean, velocity_mean = evaluate_flow(
        model, x_all, unsteady, length_scale, velocity_scale, time_scale, pressure_scale
    )

    if unsteady:
        wss_hist, ts = wss_history_unsteady(
            model, x_wall, n_wall_t, NT, length_scale, velocity_scale, time_scale
        )
        tawss = compute_tawss(wss_hist)
        osi = compute_osi(wss_hist)
        wss_mid = wss_hist[NT // 2]
        von_mises = np.sqrt(3.0) * torch.norm(wss_mid, dim=-1)

        for ti, t in enumerate(ts):
            t_col = torch.full((x_all.shape[0], 1), float(t), device=DEVICE)
            with torch.no_grad():
                field = model(build_model_input(x_all, t_col, length_scale, time_scale))
                pres_t = (field[:, 0:1] * pressure_scale).cpu().numpy()
                vel_t = (field[:, 1:4] * velocity_scale).cpu().numpy()
            wss_t = wss_hist[ti]
            wss_mag_t = torch.norm(wss_t, dim=-1).cpu().numpy()
            flow_csv = timesteps_dir / f"flow_t{ti:02d}.csv"
            np.savetxt(
                flow_csv,
                np.concatenate([xyz, pres_t, vel_t], axis=1),
                delimiter=",",
                header=f"x,y,z,p,u,v,w,time={t:.4f}s",
                comments="",
            )
            wss_csv = timesteps_dir / f"wss_t{ti:02d}.csv"
            np.savetxt(
                wss_csv,
                np.concatenate([wall_pts, wss_t.cpu().numpy(), wss_mag_t.reshape(-1, 1)], axis=1),
                delimiter=",",
                header=f"x,y,z,wss_x,wss_y,wss_z,wss_magnitude,time={t:.4f}s",
                comments="",
            )
            wall_mesh = pv.PolyData(wall_pts)
            wall_mesh["WSS"] = wss_mag_t
            wall_mesh["WSS_Vector"] = wss_t.cpu().numpy()
            wall_mesh.save(timesteps_dir / f"flow_t{ti:02d}.vtp")

        np.savetxt(
            timesteps_dir / "time_index.csv",
            np.column_stack([np.arange(NT), ts]),
            delimiter=",",
            header="timestep,time_seconds",
            comments="",
        )
    else:
        t_col_wall = torch.zeros(x_wall.shape[0], 1, device=DEVICE)
        with torch.enable_grad():
            xt_wall = build_model_input(
                x_wall, t_col_wall, length_scale, time_scale
            ).requires_grad_(True)
            field = model(xt_wall)
            wss_final = compute_wss(field, xt_wall, n_wall_t, length_scale, velocity_scale).detach()
        wss_mag = torch.norm(wss_final, dim=-1)
        tawss = wss_mag
        osi = torch.zeros_like(wss_mag)
        von_mises = np.sqrt(3.0) * wss_mag

        flow_csv = timesteps_dir / "flow_steady.csv"
        np.savetxt(
            flow_csv,
            np.concatenate([xyz, pressure_mean, velocity_mean], axis=1),
            delimiter=",",
            header="x,y,z,p,u,v,w",
            comments="",
        )
        wss_csv = timesteps_dir / "wss_steady.csv"
        np.savetxt(
            wss_csv,
            np.concatenate(
                [wall_pts, wss_final.cpu().numpy(), wss_mag.cpu().numpy().reshape(-1, 1)], axis=1
            ),
            delimiter=",",
            header="x,y,z,wss_x,wss_y,wss_z,wss_magnitude",
            comments="",
        )

    aggregate_csv = case_dir / "hemodynamics_aggregate.csv"
    tawss_np = tawss.cpu().numpy()
    osi_np = osi.cpu().numpy()
    von_mises_np = von_mises.cpu().numpy() if torch.is_tensor(von_mises) else np.asarray(von_mises)
    np.savetxt(
        aggregate_csv,
        np.concatenate(
            [wall_pts, tawss_np.reshape(-1, 1), osi_np.reshape(-1, 1), von_mises_np.reshape(-1, 1)],
            axis=1,
        ),
        delimiter=",",
        header="x,y,z,tawss,osi,von_mises",
        comments="",
    )

    wall_mesh = pv.PolyData(wall_pts)
    wall_mesh["TAWSS"] = tawss_np
    wall_mesh["OSI"] = osi_np
    wall_mesh["VonMises"] = von_mises_np
    wall_mesh.save(case_dir / "hemodynamics_aggregate.vtp")

    torch.save(model.state_dict(), case_dir / "pinn_model.pt")

    total_time = time.time() - start_time
    summary = {
        "case_name": case_name,
        **train_stats,
        "training_time_sec": float(train_time),
        "total_time_sec": float(total_time),
        "n_interior": int(len(xyz)),
        "n_wall": int(len(wall_pts)),
        "n_inlet": int(len(inlet_pts)),
        "n_outlet": int(len(outlet_pts)),
        "reynolds": float(reynolds),
        "length_scale": float(length_scale),
        "velocity_scale": float(velocity_scale),
    }
    _write_training_logs(case_name, history, summary)
    logger.info("Completed %s in %.1fs (train %.1fs)", case_name, total_time, train_time)
    return summary


# Main
def main():
    global VTP_DIR, OUTPUT_DIR, STEADY_EPOCHS, UNSTEADY_EPOCHS, UNSTEADY, LIMIT

    parser = argparse.ArgumentParser(description="Train PINN flow simulation pipeline (v12)")
    parser.add_argument(
        "--vtp-dir", default=VTP_DIR, help="Directory containing VTP geometry files"
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR, help="Output directory for PINN predictions"
    )
    parser.add_argument(
        "--steady-epochs", type=int, default=STEADY_EPOCHS, help="Steady PINN epochs"
    )
    parser.add_argument(
        "--unsteady-epochs", type=int, default=UNSTEADY_EPOCHS, help="Unsteady PINN epochs"
    )
    parser.add_argument(
        "--unsteady", action="store_true", default=UNSTEADY, help="Enable unsteady mode"
    )
    parser.add_argument("--limit", type=int, default=LIMIT, help="Limit number of VTP cases")
    args = parser.parse_args()

    VTP_DIR = args.vtp_dir
    OUTPUT_DIR = args.output_dir
    STEADY_EPOCHS = args.steady_epochs
    UNSTEADY_EPOCHS = args.unsteady_epochs
    UNSTEADY = args.unsteady
    LIMIT = args.limit

    logger.info(
        "Device: %s | unsteady=%s | epochs(steady/unsteady)=%d/%d | NT=%d | LR=%.2e",
        DEVICE,
        UNSTEADY,
        STEADY_EPOCHS,
        UNSTEADY_EPOCHS,
        NT,
        LR,
    )
    vtp_dir = Path(VTP_DIR)
    if not vtp_dir.exists():
        logger.error("VTP directory not found: %s", vtp_dir)
        sys.exit(1)

    vtp_files = sorted(vtp_dir.glob("*.vtp"))
    if LIMIT:
        vtp_files = vtp_files[:LIMIT]
    if not vtp_files:
        logger.error("No VTP files found in %s", vtp_dir)
        sys.exit(1)
    logger.info("Found %d VTP files", len(vtp_files))

    results, failed = [], []
    for vtp_file in vtp_files:
        try:
            results.append(process_case(vtp_file, Path(OUTPUT_DIR), UNSTEADY))
        except Exception as exc:
            logger.exception("Failed %s: %s", vtp_file.stem, exc)
            failed.append({"case": vtp_file.stem, "error": str(exc)})

    logger.info("Successful: %d/%d | Failed: %d", len(results), len(vtp_files), len(failed))
    if results:
        summary_path = Path(OUTPUT_DIR) / "processing_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        logger.info("Summary: %s", summary_path)
    if failed:
        fail_path = Path(OUTPUT_DIR) / "failed_cases.csv"
        with open(fail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "error"])
            w.writeheader()
            w.writerows(failed)
        logger.info("Failures: %s", fail_path)


if __name__ == "__main__":
    main()
