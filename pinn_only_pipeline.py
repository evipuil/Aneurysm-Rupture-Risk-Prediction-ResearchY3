# Version 6 source snapshot
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
from scipy.spatial import KDTree
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

HAS_PYVISTA = True
# Configuration
VTP_DIR = "vtp_data"
OUTPUT_DIR = "predictions/optimized_pinn"
FLOW_RATE = 0.2
TRAINING_LOG_SUBDIR = "training_logs"

STEADY_EPOCHS = 300
UNSTEADY_EPOCHS = 500
NT = 10
DEFAULT_EPOCHS = 1000
DEFAULT_LR = 1e-3
DEFAULT_N_INTERIOR = 4096
DEFAULT_N_WALL = 2048
DEFAULT_N_SUP = 4096
T_END = 1.0
MU = 0.0035
RHO = 1060.0
NU = MU / RHO

WALL_LOSS_WEIGHT = 1e-1
INLET_LOSS_WEIGHT = 1.0
CORRECTION_LOSS_WEIGHT = 1e-4
LOG_EVERY = 25
GRAD_CLIP_NORM = 0.0

LIMIT = None
SEED = 42
UNSTEADY = True
CUDA_DEVICE_INDEX = 0
REQUIRE_GPU = True


def _env_int(name: str, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LIMIT = _env_int("PIPELINE_LIMIT", LIMIT) if LIMIT is None else LIMIT
UNSTEADY_EPOCHS = _env_int("PIPELINE_UNSTEADY_EPOCHS", UNSTEADY_EPOCHS)
STEADY_EPOCHS = _env_int("PIPELINE_STEADY_EPOCHS", STEADY_EPOCHS)
DEFAULT_N_INTERIOR = _env_int("PIPELINE_N_INTERIOR", DEFAULT_N_INTERIOR)
DEFAULT_N_WALL = _env_int("PIPELINE_N_WALL", DEFAULT_N_WALL)
DEFAULT_N_SUP = _env_int("PIPELINE_N_SUP", DEFAULT_N_SUP)
NT = _env_int("PIPELINE_NT", NT)
DEFAULT_LR = _env_float("PIPELINE_LR", DEFAULT_LR)
FLOW_RATE = _env_float("PIPELINE_FLOW_RATE", FLOW_RATE)
WALL_LOSS_WEIGHT = _env_float("PIPELINE_WALL_LOSS_WEIGHT", WALL_LOSS_WEIGHT)
INLET_LOSS_WEIGHT = _env_float("PIPELINE_INLET_LOSS_WEIGHT", INLET_LOSS_WEIGHT)
CORRECTION_LOSS_WEIGHT = _env_float("PIPELINE_CORRECTION_LOSS_WEIGHT", CORRECTION_LOSS_WEIGHT)
UNSTEADY = _env_bool("PIPELINE_UNSTEADY", UNSTEADY)
LOG_EVERY = _env_int("PIPELINE_LOG_EVERY", LOG_EVERY)
GRAD_CLIP_NORM = _env_float("PIPELINE_GRAD_CLIP_NORM", GRAD_CLIP_NORM)
CUDA_DEVICE_INDEX = _env_int("PIPELINE_CUDA_DEVICE", CUDA_DEVICE_INDEX)
REQUIRE_GPU = _env_bool("PIPELINE_REQUIRE_GPU", REQUIRE_GPU)


def _resolve_device(require_gpu: bool, cuda_device_index: int) -> torch.device:
    if not torch.cuda.is_available():
        if require_gpu:
            raise RuntimeError(
                "CUDA is not available. This pipeline is configured to require a GPU. "
                "Set PIPELINE_REQUIRE_GPU=0 to allow CPU fallback."
            )
        return torch.device("cpu")

    n_cuda = torch.cuda.device_count()
    if cuda_device_index < 0 or cuda_device_index >= n_cuda:
        raise ValueError(
            f"Invalid PIPELINE_CUDA_DEVICE={cuda_device_index}. "
            f"Available CUDA device indices: 0..{n_cuda - 1}."
        )

    torch.cuda.set_device(cuda_device_index)
    return torch.device(f"cuda:{cuda_device_index}")


DEVICE = _resolve_device(REQUIRE_GPU, CUDA_DEVICE_INDEX)
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAINING_LOG_DIR = Path(OUTPUT_DIR) / TRAINING_LOG_SUBDIR
TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
# Logging setup
logger = logging.getLogger("PINNOnlyPipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_formatter)
logger.addHandler(_console)

_pinn_log = logging.FileHandler(Path(OUTPUT_DIR) / "pinn_correction.log")
_pinn_log.setFormatter(_formatter)
logger.addHandler(_pinn_log)


# Geometry and I/O helpers
def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


class CorrectionPINN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(4, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pulsatile_scale(t: torch.Tensor, t_end: float = T_END) -> torch.Tensor:
    return 1.0 + 0.6 * torch.sin(2 * np.pi * t / t_end)


def compute_normals(points, k=10):
    if len(points) < 3:
        centre = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
        return centre, np.array([0.0, 0.0, 1.0])

    k = min(k, len(points) - 1)
    tree = KDTree(points)
    _, indices = tree.query(points, k=k + 1)
    normals = []

    for i in range(len(points)):
        neigh = points[indices[i]]
        centered = neigh - neigh.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        _, eigenvectors = np.linalg.eigh(cov)
        n = eigenvectors[:, 0]
        normals.append(n / (np.linalg.norm(n) + 1e-12))

    return np.mean(points, axis=0), np.mean(normals, axis=0)


def extract_boundary_loops(mesh):
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if edges.n_points == 0:
        return []
    conn = edges.connectivity()
    labels = np.unique(conn["RegionId"])
    return [conn.threshold([lb - 0.1, lb + 0.1]) for lb in labels]


def build_inlet_cap(loop):
    if loop.n_points < 3:
        return loop
    try:
        return pv.PolyData(loop.points).delaunay_2d()
    except Exception:
        return loop


def pick_inlet(surface):
    loops = extract_boundary_loops(surface)
    if len(loops) == 0:
        logger.warning("No boundary loops found, using boundary edges as inlet")
        edges = surface.extract_feature_edges(boundary_edges=True)
        inlet_cap = pv.PolyData(edges.points if edges.n_points > 0 else surface.points[:100])
    else:
        caps = [build_inlet_cap(lp) for lp in loops]
        inlet_cap = max(
            caps, key=lambda c: c.area if hasattr(c, "area") and c.area > 0 else c.n_points
        )

    centre, normal = compute_normals(inlet_cap.points, k=10)
    vol_center = surface.points.mean(axis=0)
    if np.dot(normal, vol_center - centre) < 0:
        normal = -normal

    inlet_area = (
        float(inlet_cap.area)
        if hasattr(inlet_cap, "area") and inlet_cap.area > 0
        else float(max(inlet_cap.n_points, 1))
    )
    inlet_speed = FLOW_RATE / max(inlet_area, 1e-8)
    inlet_target = (inlet_speed * normal).astype(np.float32)
    return inlet_cap, inlet_target


def generate_interior_points(surface_mesh, n_points):
    logger.info("Generating %d interior points ...", n_points)
    try:
        filled = surface_mesh.fill_holes(hole_size=1000.0)
    except Exception:
        filled = surface_mesh

    bounds = filled.bounds
    n_cand = n_points * 20
    xs = np.random.uniform(bounds[0], bounds[1], n_cand)
    ys = np.random.uniform(bounds[2], bounds[3], n_cand)
    zs = np.random.uniform(bounds[4], bounds[5], n_cand)
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
        logger.info("Using distance-based fallback for interior points")
        tree = KDTree(filled.points)
        dists, _ = tree.query(candidates, k=1)
        interior = candidates[dists < np.percentile(dists, 50)]

    if len(interior) < n_points:
        logger.warning("Only %d interior pts found (wanted %d)", len(interior), n_points)
        if len(interior) < n_points // 2:
            n_extra = min(n_points - len(interior), len(filled.points))
            extra_idx = np.random.choice(len(filled.points), n_extra, replace=False)
            interior = (
                np.vstack([interior, filled.points[extra_idx]])
                if len(interior) > 0
                else filled.points[extra_idx]
            )
    else:
        idx = np.random.choice(len(interior), n_points, replace=False)
        interior = interior[idx]

    logger.info("Final interior points: %d", len(interior))
    return interior.astype(np.float32)


def extract_wall_points(vtp_file: Path, n_wall_points: int = DEFAULT_N_WALL):
    mesh = pv.read(vtp_file) if HAS_PYVISTA else None
    if mesh is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    surface = mesh.extract_surface() if isinstance(mesh, pv.UnstructuredGrid) else mesh

    if surface.point_normals is None:
        surface = surface.compute_normals(point_normals=True, cell_normals=False)

    wall_points = surface.points.astype(np.float32)
    wall_normals = surface.point_normals.astype(np.float32)

    if len(wall_points) > n_wall_points:
        idx = np.random.choice(len(wall_points), n_wall_points, replace=False)
        wall_points = wall_points[idx]
        wall_normals = wall_normals[idx]

    return wall_points, wall_normals


def compute_wss(
    velocity: torch.Tensor, xt_wall: torch.Tensor, wall_normals: torch.Tensor, mu: float = MU
):
    if velocity.shape[0] != xt_wall.shape[0] or velocity.shape[0] != wall_normals.shape[0]:
        raise ValueError(
            "WSS shape mismatch: velocity=%s, XT_wall=%s, wall_normals=%s"
            % (velocity.shape, xt_wall.shape, wall_normals.shape)
        )

    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    w = velocity[:, 2:3]

    u_g = gradients(u, xt_wall)
    v_g = gradients(v, xt_wall)
    w_g = gradients(w, xt_wall)

    u_n = (
        u_g[:, 0:1] * wall_normals[:, 0:1]
        + u_g[:, 1:2] * wall_normals[:, 1:2]
        + u_g[:, 2:3] * wall_normals[:, 2:3]
    )
    v_n = (
        v_g[:, 0:1] * wall_normals[:, 0:1]
        + v_g[:, 1:2] * wall_normals[:, 1:2]
        + v_g[:, 2:3] * wall_normals[:, 2:3]
    )
    w_n = (
        w_g[:, 0:1] * wall_normals[:, 0:1]
        + w_g[:, 1:2] * wall_normals[:, 1:2]
        + w_g[:, 2:3] * wall_normals[:, 2:3]
    )

    return mu * torch.cat([u_n, v_n, w_n], dim=1)


def compute_wss_magnitude(wss_vector: torch.Tensor) -> torch.Tensor:
    return torch.norm(wss_vector, dim=-1)


def compute_osi(wss_history):
    wss_stack = torch.stack(wss_history, dim=0)
    wss_avg = wss_stack.mean(dim=0)
    wss_mag_avg = torch.norm(wss_stack, dim=-1).mean(dim=0)
    wss_avg_mag = torch.norm(wss_avg, dim=-1)
    return 0.5 * (1.0 - wss_avg_mag / (wss_mag_avg + 1e-8))


def compute_von_mises_stress(wss_vector: torch.Tensor):
    wss_mag = torch.norm(wss_vector, dim=-1)
    return np.sqrt(3.0) * wss_mag


def compute_tawss(wss_history):
    wss_mags = torch.stack([torch.norm(wss, dim=-1) for wss in wss_history], dim=0)
    return wss_mags.mean(dim=0)


def _write_case_training_logs(case_name: str, history: list, summary: dict):
    epoch_path = TRAINING_LOG_DIR / f"{case_name}_epoch_losses.csv"
    if history:
        fields = [
            "epoch",
            "total_loss",
            "physics_loss",
            "physics_norm",
            "wall_loss",
            "inlet_loss",
            "correction_loss",
            "lr",
            "epoch_time_sec",
        ]
        with open(epoch_path, "w", newline="") as f:
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


# PINN training
def train_steady_pinn_correction(
    x_int,
    x_inlet,
    x_wall,
    inlet_target,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
):
    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200
    )

    x_int = x_int.to(DEVICE).requires_grad_(True)
    x_inlet = x_inlet.to(DEVICE)
    x_wall = x_wall.to(DEVICE)
    inlet_target = torch.tensor(inlet_target, dtype=torch.float32, device=DEVICE).view(1, 3)

    logger.info(
        "Training steady PINN: %d epochs, %d interior pts, %d inlet pts, %d wall pts",
        epochs,
        x_int.shape[0],
        x_inlet.shape[0],
        x_wall.shape[0],
    )
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="PINN Correction Training (steady)")
    last_loss_physics = 0.0
    last_loss_wall = 0.0
    last_loss_inlet = 0.0
    physics_ref = None
    history = []
    best_epoch = 0
    wall_point_factor = max(1.0, float(x_int.shape[0]) / max(1.0, float(x_wall.shape[0])))
    inlet_point_factor = max(1.0, float(x_int.shape[0]) / max(1.0, float(x_inlet.shape[0])))

    for epoch in pbar:
        t_epoch = time.time()
        optimizer.zero_grad()

        t_zeros_int = torch.zeros((x_int.shape[0], 1), device=DEVICE)
        xt_int = torch.cat([x_int, t_zeros_int], dim=1).requires_grad_(True)
        field = model(xt_int)

        p = field[:, 0:1]
        u = field[:, 1:2]
        v = field[:, 2:3]
        w = field[:, 3:4]

        u_x = gradients(u, xt_int)[:, 0:1]
        u_y = gradients(u, xt_int)[:, 1:2]
        u_z = gradients(u, xt_int)[:, 2:3]
        v_x = gradients(v, xt_int)[:, 0:1]
        v_y = gradients(v, xt_int)[:, 1:2]
        v_z = gradients(v, xt_int)[:, 2:3]
        w_x = gradients(w, xt_int)[:, 0:1]
        w_y = gradients(w, xt_int)[:, 1:2]
        w_z = gradients(w, xt_int)[:, 2:3]
        p_x = gradients(p, xt_int)[:, 0:1]
        p_y = gradients(p, xt_int)[:, 1:2]
        p_z = gradients(p, xt_int)[:, 2:3]

        u_xx = gradients(u_x, xt_int)[:, 0:1]
        u_yy = gradients(u_y, xt_int)[:, 1:2]
        u_zz = gradients(u_z, xt_int)[:, 2:3]
        v_xx = gradients(v_x, xt_int)[:, 0:1]
        v_yy = gradients(v_y, xt_int)[:, 1:2]
        v_zz = gradients(v_z, xt_int)[:, 2:3]
        w_xx = gradients(w_x, xt_int)[:, 0:1]
        w_yy = gradients(w_y, xt_int)[:, 1:2]
        w_zz = gradients(w_z, xt_int)[:, 2:3]

        mom_u = (u * u_x + v * u_y + w * u_z) + p_x - NU * (u_xx + u_yy + u_zz)
        mom_v = (u * v_x + v * v_y + w * v_z) + p_y - NU * (v_xx + v_yy + v_zz)
        mom_w = (u * w_x + v * w_y + w * w_z) + p_z - NU * (w_xx + w_yy + w_zz)
        cont = u_x + v_y + w_z

        loss_physics = (
            mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
        )

        t_zeros_inlet = torch.zeros((x_inlet.shape[0], 1), device=DEVICE)
        xt_inlet = torch.cat([x_inlet, t_zeros_inlet], dim=1)
        field_inlet = model(xt_inlet)
        loss_inlet = (field_inlet[:, 1:4] - inlet_target).pow(2).mean() * inlet_point_factor

        t_zeros_wall = torch.zeros((x_wall.shape[0], 1), device=DEVICE)
        xt_wall = torch.cat([x_wall, t_zeros_wall], dim=1)
        field_wall = model(xt_wall)
        loss_wall = field_wall[:, 1:4].pow(2).mean() * wall_point_factor

        loss_correction = field.pow(2).mean()
        if physics_ref is None:
            physics_ref = float(loss_physics.item()) + 1e-12

        loss_physics_norm = loss_physics / physics_ref
        loss_wall_norm = loss_wall
        loss = (
            loss_physics_norm
            + WALL_LOSS_WEIGHT * loss_wall_norm
            + INLET_LOSS_WEIGHT * loss_inlet
            + CORRECTION_LOSS_WEIGHT * loss_correction
        )

        loss.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step(loss)

        last_loss_physics = float(loss_physics.item())
        last_loss_wall = float(loss_wall.item())
        last_loss_inlet = float(loss_inlet.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch + 1

        epoch_time = time.time() - t_epoch
        history.append(
            {
                "epoch": epoch + 1,
                "total_loss": float(loss.item()),
                "physics_loss": float(loss_physics.item()),
                "physics_norm": float(loss_physics_norm.item()),
                "wall_loss": float(loss_wall.item()),
                "inlet_loss": float(loss_inlet.item()),
                "correction_loss": float(loss_correction.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": float(epoch_time),
            }
        )

        if epoch % 100 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3e}",
                    "phys": f"{loss_physics.item():.3e}",
                    "wall": f"{loss_wall.item():.3e}",
                    "inlet": f"{loss_inlet.item():.3e}",
                }
            )
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                "[steady] epoch %d/%d | total=%.3e phys=%.3e wall=%.3e inlet=%.3e corr=%.3e | phys_n=%.3e wall_n=%.3e",
                epoch + 1,
                epochs,
                loss.item(),
                loss_physics.item(),
                loss_wall.item(),
                loss_inlet.item(),
                loss_correction.item(),
                loss_physics_norm.item(),
                loss_wall_norm.item(),
            )

    logger.info(
        "Steady PINN training complete. Best loss: %.3e | final phys=%.3e wall=%.3e inlet=%.3e",
        best_loss,
        last_loss_physics,
        last_loss_wall,
        last_loss_inlet,
    )
    train_stats = {
        "mode": "steady",
        "best_epoch": int(best_epoch),
        "best_total_loss": float(best_loss),
        "final_physics_loss": float(last_loss_physics),
        "final_wall_loss": float(last_loss_wall),
        "final_inlet_loss": float(last_loss_inlet),
        "epochs_completed": int(len(history)),
        "avg_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history]))
        if history
        else 0.0,
        "min_epoch_time_sec": float(np.min([h["epoch_time_sec"] for h in history]))
        if history
        else 0.0,
        "max_epoch_time_sec": float(np.max([h["epoch_time_sec"] for h in history]))
        if history
        else 0.0,
    }
    return model, history, train_stats


def train_unsteady_pinn_correction(
    x_int,
    x_inlet,
    x_wall,
    inlet_target,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    nt=NT,
):
    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_int = x_int.to(DEVICE)
    x_inlet = x_inlet.to(DEVICE)
    x_wall = x_wall.to(DEVICE)
    inlet_target = torch.tensor(inlet_target, dtype=torch.float32, device=DEVICE).view(1, 3)

    t_vals = torch.linspace(0, T_END, nt, device=DEVICE)
    logger.info("Training unsteady PINN: %d epochs, %d time steps", epochs, nt)
    pbar = tqdm(range(epochs), desc="PINN Correction Training (unsteady)")
    best_total = float("inf")
    last_loss_physics = 0.0
    last_loss_wall = 0.0
    last_loss_inlet = 0.0
    physics_ref = None
    history = []
    best_epoch = 0
    wall_point_factor = max(1.0, float(x_int.shape[0]) / max(1.0, float(x_wall.shape[0])))
    inlet_point_factor = max(1.0, float(x_int.shape[0]) / max(1.0, float(x_inlet.shape[0])))

    for epoch in pbar:
        t_epoch = time.time()
        optimizer.zero_grad()
        loss_physics = 0.0
        loss_wall = 0.0
        loss_inlet = 0.0
        loss_correction = 0.0

        for t in t_vals:
            t_col = t * torch.ones((x_int.shape[0], 1), device=DEVICE)
            xt_int = torch.cat([x_int, t_col], dim=1).requires_grad_(True)
            scale = pulsatile_scale(t)
            field = model(xt_int)

            p = field[:, 0:1]
            u = field[:, 1:2]
            v = field[:, 2:3]
            w = field[:, 3:4]

            u_t = gradients(u, xt_int)[:, 3:4]
            v_t = gradients(v, xt_int)[:, 3:4]
            w_t = gradients(w, xt_int)[:, 3:4]
            u_x = gradients(u, xt_int)[:, 0:1]
            u_y = gradients(u, xt_int)[:, 1:2]
            u_z = gradients(u, xt_int)[:, 2:3]
            v_x = gradients(v, xt_int)[:, 0:1]
            v_y = gradients(v, xt_int)[:, 1:2]
            v_z = gradients(v, xt_int)[:, 2:3]
            w_x = gradients(w, xt_int)[:, 0:1]
            w_y = gradients(w, xt_int)[:, 1:2]
            w_z = gradients(w, xt_int)[:, 2:3]
            p_x = gradients(p, xt_int)[:, 0:1]
            p_y = gradients(p, xt_int)[:, 1:2]
            p_z = gradients(p, xt_int)[:, 2:3]

            u_xx = gradients(u_x, xt_int)[:, 0:1]
            u_yy = gradients(u_y, xt_int)[:, 1:2]
            u_zz = gradients(u_z, xt_int)[:, 2:3]
            v_xx = gradients(v_x, xt_int)[:, 0:1]
            v_yy = gradients(v_y, xt_int)[:, 1:2]
            v_zz = gradients(v_z, xt_int)[:, 2:3]
            w_xx = gradients(w_x, xt_int)[:, 0:1]
            w_yy = gradients(w_y, xt_int)[:, 1:2]
            w_zz = gradients(w_z, xt_int)[:, 2:3]

            mom_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - NU * (u_xx + u_yy + u_zz)
            mom_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - NU * (v_xx + v_yy + v_zz)
            mom_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - NU * (w_xx + w_yy + w_zz)
            cont = u_x + v_y + w_z

            loss_physics += (
                mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
            )

            t_col_inlet = t * torch.ones((x_inlet.shape[0], 1), device=DEVICE)
            xt_inlet = torch.cat([x_inlet, t_col_inlet], dim=1)
            field_inlet = model(xt_inlet)
            loss_inlet += (field_inlet[:, 1:4] - scale * inlet_target).pow(
                2
            ).mean() * inlet_point_factor

            t_col_wall = t * torch.ones((x_wall.shape[0], 1), device=DEVICE)
            xt_wall = torch.cat([x_wall, t_col_wall], dim=1)
            field_wall = model(xt_wall)
            loss_wall += field_wall[:, 1:4].pow(2).mean() * wall_point_factor

            loss_correction += field.pow(2).mean()

        loss_physics /= nt
        loss_wall /= nt
        loss_inlet /= nt
        loss_correction /= nt
        if physics_ref is None:
            physics_ref = float(loss_physics.item()) + 1e-12

        loss_physics_norm = loss_physics / physics_ref
        loss_wall_norm = loss_wall
        loss = (
            loss_physics_norm
            + WALL_LOSS_WEIGHT * loss_wall_norm
            + INLET_LOSS_WEIGHT * loss_inlet
            + CORRECTION_LOSS_WEIGHT * loss_correction
        )

        loss.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        cur_total = float(loss.item())
        if cur_total < best_total:
            best_total = cur_total
            best_epoch = epoch + 1
        last_loss_physics = float(loss_physics.item())
        last_loss_wall = float(loss_wall.item())
        last_loss_inlet = float(loss_inlet.item())

        epoch_time = time.time() - t_epoch
        history.append(
            {
                "epoch": epoch + 1,
                "total_loss": float(loss.item()),
                "physics_loss": float(loss_physics.item()),
                "physics_norm": float(loss_physics_norm.item()),
                "wall_loss": float(loss_wall.item()),
                "inlet_loss": float(loss_inlet.item()),
                "correction_loss": float(loss_correction.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": float(epoch_time),
            }
        )

        if epoch % 100 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3e}",
                    "phys": f"{loss_physics.item():.3e}",
                    "wall": f"{loss_wall.item():.3e}",
                    "inlet": f"{loss_inlet.item():.3e}",
                }
            )
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                "[unsteady] epoch %d/%d | total=%.3e phys=%.3e wall=%.3e inlet=%.3e corr=%.3e | phys_n=%.3e wall_n=%.3e",
                epoch + 1,
                epochs,
                loss.item(),
                loss_physics.item(),
                loss_wall.item(),
                loss_inlet.item(),
                loss_correction.item(),
                loss_physics_norm.item(),
                loss_wall_norm.item(),
            )

    logger.info(
        "Unsteady PINN training complete. Best loss: %.3e | final phys=%.3e wall=%.3e inlet=%.3e",
        best_total,
        last_loss_physics,
        last_loss_wall,
        last_loss_inlet,
    )
    train_stats = {
        "mode": "unsteady",
        "best_epoch": int(best_epoch),
        "best_total_loss": float(best_total),
        "final_physics_loss": float(last_loss_physics),
        "final_wall_loss": float(last_loss_wall),
        "final_inlet_loss": float(last_loss_inlet),
        "epochs_completed": int(len(history)),
        "avg_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history]))
        if history
        else 0.0,
        "min_epoch_time_sec": float(np.min([h["epoch_time_sec"] for h in history]))
        if history
        else 0.0,
        "max_epoch_time_sec": float(np.max([h["epoch_time_sec"] for h in history]))
        if history
        else 0.0,
    }
    return model, history, train_stats


# End-to-end case processing
def process_single_case(
    vtp_file: Path,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    n_interior: int = DEFAULT_N_INTERIOR,
    n_wall: int = DEFAULT_N_WALL,
    unsteady: bool = True,
):
    case_name = vtp_file.stem
    start_time = time.time()
    logger.info("Processing PINN correction: %s", case_name)

    mesh = pv.read(vtp_file)
    surface = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
    xyz = generate_interior_points(surface, n_interior)
    wall_points, wall_normals = extract_wall_points(vtp_file, n_wall)
    inlet_cap, inlet_target = pick_inlet(surface)
    logger.info("Using %d interior points for flow field output", len(xyz))

    x_all = torch.tensor(xyz, dtype=torch.float32, device=DEVICE)

    x_wall = torch.tensor(wall_points, dtype=torch.float32, device=DEVICE)
    n_wall_t = torch.tensor(wall_normals, dtype=torch.float32, device=DEVICE)
    inlet_points = inlet_cap.points.astype(np.float32)
    if len(inlet_points) > DEFAULT_N_SUP:
        inlet_points = inlet_points[
            np.random.choice(len(inlet_points), DEFAULT_N_SUP, replace=False)
        ]
    x_inlet = torch.tensor(inlet_points, dtype=torch.float32, device=DEVICE)

    n_points = len(xyz)
    n_int = min(n_points, n_interior)
    idx_int = np.linspace(0, n_points - 1, n_int, dtype=int)

    x_int = x_all[idx_int]

    logger.info("Training PINN from VTP geometry only...")
    train_start = time.time()

    if unsteady:
        model, loss_history, train_stats = train_unsteady_pinn_correction(
            x_int,
            x_inlet,
            x_wall,
            inlet_target,
            epochs=epochs,
            lr=DEFAULT_LR,
            nt=NT,
        )
    else:
        model, loss_history, train_stats = train_steady_pinn_correction(
            x_int,
            x_inlet,
            x_wall,
            inlet_target,
            epochs=epochs,
            lr=DEFAULT_LR,
        )

    train_time = time.time() - train_start
    logger.info("PINN training time: %.1fs", train_time)
    case_summary = {
        "case_name": case_name,
        "mode": train_stats["mode"],
        "epochs_completed": train_stats["epochs_completed"],
        "best_epoch": train_stats["best_epoch"],
        "best_total_loss": train_stats["best_total_loss"],
        "final_physics_loss": train_stats["final_physics_loss"],
        "final_wall_loss": train_stats["final_wall_loss"],
        "final_inlet_loss": train_stats["final_inlet_loss"],
        "avg_epoch_time_sec": train_stats["avg_epoch_time_sec"],
        "min_epoch_time_sec": train_stats["min_epoch_time_sec"],
        "max_epoch_time_sec": train_stats["max_epoch_time_sec"],
        "training_time_sec": float(train_time),
        "n_output_points": int(n_points),
        "n_wall_points": int(len(wall_points)),
        "n_inlet_points": int(x_inlet.shape[0]),
    }
    _write_case_training_logs(case_name, loss_history, case_summary)

    model.eval()
    with torch.no_grad():
        if unsteady:
            velocity_accum = torch.zeros((x_all.shape[0], 3), device=DEVICE)
            pressure_accum = torch.zeros((x_all.shape[0], 1), device=DEVICE)
            t_eval = torch.linspace(0, T_END, steps=10, device=DEVICE)
            for t_val in t_eval:
                t_col = t_val * torch.ones((x_all.shape[0], 1), device=DEVICE)
                xt_all = torch.cat([x_all, t_col], dim=1)
                field_all = model(xt_all)
                pressure_accum += field_all[:, 0:1]
                velocity_accum += field_all[:, 1:4]
            velocity_corrected = (velocity_accum / len(t_eval)).cpu().numpy()
            pressure_corrected = (pressure_accum / len(t_eval)).cpu().numpy()
        else:
            t_col = torch.zeros((x_all.shape[0], 1), device=DEVICE)
            xt_all = torch.cat([x_all, t_col], dim=1)
            field_all = model(xt_all)
            pressure_corrected = field_all[:, 0:1].cpu().numpy()
            velocity_corrected = field_all[:, 1:4].cpu().numpy()

    if unsteady:
        wss_history = []
        t_vals = np.linspace(0, T_END, NT)
        for t in t_vals:
            t_col_wall = torch.full((x_wall.shape[0], 1), t, device=DEVICE)

            with torch.enable_grad():
                xt_wall = torch.cat([x_wall, t_col_wall], dim=1).requires_grad_(True)
                field_wall = model(xt_wall)
                vel_wall = field_wall[:, 1:4]
                wss = compute_wss(vel_wall, xt_wall, n_wall_t)
            wss_history.append(wss.detach())

        wss_final = wss_history[NT // 2]
        tawss = compute_tawss(wss_history)
        osi = compute_osi(wss_history)
        von_mises = compute_von_mises_stress(wss_final)
    else:
        t_col_wall = torch.zeros((x_wall.shape[0], 1), device=DEVICE)
        with torch.enable_grad():
            xt_wall = torch.cat([x_wall, t_col_wall], dim=1).requires_grad_(True)
            field_wall = model(xt_wall)
            vel_wall = field_wall[:, 1:4]
            wss_final = compute_wss(vel_wall, xt_wall, n_wall_t)

        wss_final = wss_final.detach()
        wss_mag = compute_wss_magnitude(wss_final)
        von_mises = compute_von_mises_stress(wss_final)
        tawss = wss_mag
        osi = torch.zeros_like(wss_mag)

    case_dir = Path(output_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    timesteps_dir = case_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)

    if unsteady:
        t_vals = np.linspace(0, T_END, NT)
        for ti, t in enumerate(t_vals):
            t_col_all = torch.full((x_all.shape[0], 1), t, device=DEVICE)
            t_col_wall = torch.full((x_wall.shape[0], 1), t, device=DEVICE)

            with torch.no_grad():
                xt_all = torch.cat([x_all, t_col_all], dim=1)
                field_all = model(xt_all)
                pres_t = field_all[:, 0:1]
                vel_t = field_all[:, 1:4]

            with torch.enable_grad():
                xt_wall = torch.cat([x_wall, t_col_wall], dim=1).requires_grad_(True)
                field_wall = model(xt_wall)
                vel_wall = field_wall[:, 1:4]
                wss_t = compute_wss(vel_wall, xt_wall, n_wall_t)

            wss_t = wss_t.detach()
            wss_mag_t = compute_wss_magnitude(wss_t)
            velocity_t = vel_t.detach().cpu().numpy()
            pressure_t = pres_t.detach().cpu().numpy()

            flow_csv = timesteps_dir / f"flow_t{ti:02d}.csv"
            flow_data = np.concatenate([xyz, pressure_t, velocity_t], axis=1)
            np.savetxt(
                flow_csv,
                flow_data,
                delimiter=",",
                header=f"x,y,z,p,u,v,w,time={t:.4f}s",
                comments="",
            )

            wss_csv = timesteps_dir / f"wss_t{ti:02d}.csv"
            wss_data = np.concatenate(
                [wall_points, wss_t.cpu().numpy(), wss_mag_t.cpu().numpy().reshape(-1, 1)],
                axis=1,
            )
            np.savetxt(
                wss_csv,
                wss_data,
                delimiter=",",
                header=f"x,y,z,wss_x,wss_y,wss_z,wss_magnitude,time={t:.4f}s",
                comments="",
            )

            if HAS_PYVISTA:
                vtp_t = timesteps_dir / f"flow_t{ti:02d}.vtp"
                wall_mesh_t = pv.PolyData(wall_points)
                wall_mesh_t["WSS"] = wss_mag_t.cpu().numpy()
                wall_mesh_t["WSS_Vector"] = wss_t.cpu().numpy()
                wall_mesh_t.save(vtp_t)

        time_index = timesteps_dir / "time_index.csv"
        np.savetxt(
            time_index,
            np.column_stack([np.arange(NT), t_vals]),
            delimiter=",",
            header="timestep,time_seconds",
            comments="",
        )
    else:
        flow_csv = timesteps_dir / "flow_steady.csv"
        corrected_data = np.concatenate([xyz, pressure_corrected, velocity_corrected], axis=1)
        np.savetxt(flow_csv, corrected_data, delimiter=",", header="x,y,z,p,u,v,w", comments="")

        wss_csv = timesteps_dir / "wss_steady.csv"
        wss_mag = compute_wss_magnitude(wss_final)
        wss_data = np.concatenate(
            [wall_points, wss_final.cpu().numpy(), wss_mag.cpu().numpy().reshape(-1, 1)],
            axis=1,
        )
        np.savetxt(
            wss_csv,
            wss_data,
            delimiter=",",
            header="x,y,z,wss_x,wss_y,wss_z,wss_magnitude",
            comments="",
        )

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

    model_file = case_dir / "pinn_model.pt"
    torch.save(model.state_dict(), model_file)

    if HAS_PYVISTA:
        vtp_aggregate = case_dir / "hemodynamics_aggregate.vtp"
        wall_mesh = pv.PolyData(wall_points)
        wall_mesh["TAWSS"] = tawss.cpu().numpy()
        wall_mesh["OSI"] = osi.cpu().numpy()
        wall_mesh["VonMises"] = von_mises.cpu().numpy()
        wall_mesh.save(vtp_aggregate)

    total_time = time.time() - start_time
    logger.info("Case %s completed in %.1fs", case_name, total_time)

    return {
        "case_name": case_name,
        "n_points": n_points,
        "n_wall": len(wall_points),
        "train_time": train_time,
        "total_time": total_time,
        "output_dir": str(case_dir),
    }


# Main pipeline
def main():
    logger.info(
        "Device: %s | NT: %d | Unsteady epochs: %d | Wall loss weight: %.2f | LR: %.2e",
        DEVICE,
        NT,
        UNSTEADY_EPOCHS,
        WALL_LOSS_WEIGHT,
        DEFAULT_LR,
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
    logger.info("=" * 60)

    results = []
    failed = []
    for vtp_path in vtp_files:
        case = vtp_path.stem
        logger.info("Processing case: %s", case)
        try:
            results.append(
                process_single_case(
                    vtp_path,
                    Path(OUTPUT_DIR),
                    epochs=UNSTEADY_EPOCHS if UNSTEADY else STEADY_EPOCHS,
                    n_interior=DEFAULT_N_INTERIOR,
                    n_wall=DEFAULT_N_WALL,
                    unsteady=UNSTEADY,
                )
            )
        except Exception as exc:
            logger.error("Failed %s: %s", case, exc)
            failed.append({"case": case, "error": str(exc)})

    logger.info("=" * 60)
    logger.info("Successful: %d/%d   Failed: %d", len(results), len(vtp_files), len(failed))

    if results:
        avg_t = np.mean([r["total_time"] for r in results])
        logger.info("Avg total time: %.2fs", avg_t)

        summary = Path(OUTPUT_DIR) / "processing_summary.csv"
        with open(summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        logger.info("Summary saved to %s", summary)

    if failed:
        fail_path = Path(OUTPUT_DIR) / "failed_cases.csv"
        with open(fail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "error"])
            w.writeheader()
            w.writerows(failed)
        logger.info("Failed cases logged to %s", fail_path)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
