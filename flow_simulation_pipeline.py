# Version 6 source snapshot
import csv
import logging
import os
import sys
import time
from pathlib import Path

import neural_networks as nn_net
import numpy as np
import pyvista as pv
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy.spatial import KDTree
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Make local imports consistent with prior scripts
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "old_scripts"))

HAS_PYVISTA = True
# Configuration (merged from batch_vtp_inference.py + pinn_correction.py)
# Input / output
VTP_DIR = "vtp_data"
BATCH_OUTPUT_DIR = "predictions/batch_results"
PINN_OUTPUT_DIR = "predictions/optimized_pinn"
PREDICTIONS_DIR = BATCH_OUTPUT_DIR  # compatibility alias

# DeepONet inference config
CHECKPOINT_DIR = "cfd_opt_deeponet/checkpoint/deeponet"
CHECKPOINT_ID = 5000
N_POINTS = 10000
FLOW_RATE = 0.2
SAVE_NPY = False
SAVE_VTP = False
USE_MIXED_PRECISION = False

# PINN correction config
STEADY_EPOCHS = 300
UNSTEADY_EPOCHS = 500
NT = 10
DEFAULT_EPOCHS = 1000
DEFAULT_LR = 1e-3
DEFAULT_N_INTERIOR = 4096
DEFAULT_N_WALL = 2048
DEFAULT_N_SUP = 4096
T_END = 1.0
EARLY_STOP_PATIENCE = 100
LEARNING_RATE = 1e-3
WALL_TOLERANCE = 0.5
MU = 0.0035
RHO = 1060.0
NU = MU / RHO

# Requested change: increase wall-loss weight
WALL_LOSS_WEIGHT = 1e-1
CORRECTION_LOSS_WEIGHT = 1e-4
LOG_EVERY = 25
GRAD_CLIP_NORM = 0.0

# Runtime controls
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
N_POINTS = _env_int("PIPELINE_N_POINTS", N_POINTS)
UNSTEADY_EPOCHS = _env_int("PIPELINE_UNSTEADY_EPOCHS", UNSTEADY_EPOCHS)
STEADY_EPOCHS = _env_int("PIPELINE_STEADY_EPOCHS", STEADY_EPOCHS)
DEFAULT_N_INTERIOR = _env_int("PIPELINE_N_INTERIOR", DEFAULT_N_INTERIOR)
DEFAULT_N_WALL = _env_int("PIPELINE_N_WALL", DEFAULT_N_WALL)
DEFAULT_N_SUP = _env_int("PIPELINE_N_SUP", DEFAULT_N_SUP)
NT = _env_int("PIPELINE_NT", NT)
DEFAULT_LR = _env_float("PIPELINE_LR", DEFAULT_LR)
FLOW_RATE = _env_float("PIPELINE_FLOW_RATE", FLOW_RATE)
WALL_LOSS_WEIGHT = _env_float("PIPELINE_WALL_LOSS_WEIGHT", WALL_LOSS_WEIGHT)
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
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
os.makedirs(PINN_OUTPUT_DIR, exist_ok=True)
# Logging setup (preserves both stage log files)
logger = logging.getLogger("FlowSimulationPipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_formatter)
logger.addHandler(_console)

_batch_log = logging.FileHandler(Path(BATCH_OUTPUT_DIR) / "batch_inference.log")
_batch_log.setFormatter(_formatter)
logger.addHandler(_batch_log)

_pinn_log = logging.FileHandler(Path(PINN_OUTPUT_DIR) / "pinn_correction.log")
_pinn_log.setFormatter(_formatter)
logger.addHandler(_pinn_log)


# DeepONet geometry + inference helpers
def compute_normals(points, k=10):
    # Estimate a representative center and average normal from local PCA.
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

    # Fallback heuristic if enclosure query underperforms.
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


def preprocess_vtp(vtp_file, n_points, flow_rate):
    mesh = pv.read(vtp_file)
    surface = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh

    x_internal = generate_interior_points(surface, n_points)

    loops = extract_boundary_loops(surface)
    if len(loops) == 0:
        logger.warning("No boundary loops found, using boundary edges as inlet")
        edges = surface.extract_feature_edges(boundary_edges=True)
        inlet_cap = pv.PolyData(edges.points if edges.n_points > 0 else surface.points[:100])
    else:
        caps = [build_inlet_cap(lp) for lp in loops]
        inlet_cap = max(
            caps,
            key=lambda c: c.area if hasattr(c, "area") and c.area > 0 else c.n_points,
        )

    tree = KDTree(surface.points)
    sdf = tree.query(x_internal, k=1)[0].reshape(-1, 1)

    x_sup = np.concatenate([x_internal, sdf], axis=-1).astype(np.float32)[np.newaxis]
    y_sup = np.zeros((1, len(x_internal), 4), dtype=np.float32)
    x_inlet = inlet_cap.points.copy().astype(np.float32)[np.newaxis]

    centre, normal = compute_normals(inlet_cap.points, k=10)
    vol_center = x_internal.mean(axis=0)
    if np.dot(normal, vol_center - centre) < 0:
        normal = -normal

    simple_inlet = np.concatenate([centre, normal, [flow_rate]]).astype(np.float32).reshape(1, -1)
    return x_sup, y_sup, x_inlet, simple_inlet


def load_deeponet(checkpoint_dir, device, checkpoint_id):
    bc_dim = 64
    hidden_num = bc_dim
    out_dim = 4 * hidden_num
    layer_num = 4
    in_dim = 4

    trunk = nn_net.Trunk(in_dim, out_dim, hidden_num, layer_num).to(device)
    branch_bc = nn_net.Branch(7, bc_dim, bc_dim, 4).to(device)
    branch_bp = nn_net.Branch_Bypass(1, 4).to(device)

    def _load(net, name):
        state = torch.load(Path(checkpoint_dir) / f"{name}_{checkpoint_id}", map_location=device)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        net.load_state_dict(state)

    _load(trunk, "trunk")
    _load(branch_bc, "branch_bc")
    _load(branch_bp, "branch_bp")

    trunk.eval()
    branch_bc.eval()
    branch_bp.eval()
    return trunk, branch_bc, branch_bp


def run_inference(x_sup, simple_inlet, trunk, branch_bc, branch_bp, device):
    x = torch.tensor(x_sup, dtype=torch.float32, device=device)
    x_in = torch.tensor(simple_inlet, dtype=torch.float32, device=device)

    with torch.no_grad():
        t1, t2, t3, t4 = trunk(x)
        bc = branch_bc(x_in).unsqueeze(-1)
        h1 = torch.matmul(t1, bc)
        h2 = torch.matmul(t2, bc)
        h3 = torch.matmul(t3, bc)
        h4 = torch.matmul(t4, bc)
        y = torch.cat([h1, h2, h3, h4], dim=-1)
        bp = branch_bp(x_in[..., -1:])
        y = y * bp.unsqueeze(1)

    return y[0].cpu().numpy()  # (N, 4) -> [p, u, v, w]


def save_csv(points, preds, path):
    data = np.concatenate([points, preds], axis=1)
    np.savetxt(path, data, delimiter=",", header="x,y,z,p,u,v,w", comments="")


def save_npy(points, preds, path):
    np.save(path, np.concatenate([points, preds], axis=1))


def save_vtp(points, preds, path):
    cloud = pv.PolyData(points)
    cloud["Pressure"] = preds[:, 0]
    cloud["Velocity"] = preds[:, 1:4]
    cloud["VelocityMagnitude"] = np.linalg.norm(preds[:, 1:4], axis=1)
    cloud.save(path)


# PINN correction helpers
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


def train_steady_pinn_correction(
    x_int,
    y_int_baseline,
    x_sup,
    y_sup_baseline,
    x_wall,
    y_wall_baseline,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
):
    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200
    )

    x_int = x_int.to(DEVICE).requires_grad_(True)
    y_int_baseline = y_int_baseline.to(DEVICE)
    x_sup = x_sup.to(DEVICE)
    y_sup_baseline = y_sup_baseline.to(DEVICE)
    x_wall = x_wall.to(DEVICE)
    y_wall_baseline = y_wall_baseline.to(DEVICE)

    logger.info(
        "Training steady correction PINN: %d epochs, %d interior pts, %d wall pts",
        epochs,
        x_int.shape[0],
        x_wall.shape[0],
    )
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="PINN Correction Training (steady)")
    last_loss_physics = 0.0
    last_loss_wall = 0.0
    physics_ref = None
    vel_scale = y_int_baseline[:, 0:3].abs().max().detach().clamp_min(1e-8)
    wall_point_factor = max(1.0, float(x_int.shape[0]) / max(1.0, float(x_wall.shape[0])))

    for epoch in pbar:
        optimizer.zero_grad()

        t_zeros_int = torch.zeros((x_int.shape[0], 1), device=DEVICE)
        xt_int = torch.cat([x_int, t_zeros_int], dim=1).requires_grad_(True)
        delta = model(xt_int)

        du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]
        u = y_int_baseline[:, 0:1] + du
        v = y_int_baseline[:, 1:2] + dv
        w = y_int_baseline[:, 2:3] + dw
        p = y_int_baseline[:, 3:4] + dp

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

        t_zeros_wall = torch.zeros((x_wall.shape[0], 1), device=DEVICE)
        xt_wall = torch.cat([x_wall, t_zeros_wall], dim=1)
        delta_wall = model(xt_wall)
        wall_target = -y_wall_baseline[:, 0:3] / vel_scale
        wall_pred = delta_wall[:, 0:3] / vel_scale
        loss_wall = (wall_pred - wall_target).pow(2).mean() * wall_point_factor

        loss_correction = du.pow(2).mean() + dv.pow(2).mean() + dw.pow(2).mean() + dp.pow(2).mean()
        if physics_ref is None:
            physics_ref = float(loss_physics.item()) + 1e-12

        loss_physics_norm = loss_physics / physics_ref
        loss_wall_norm = loss_wall
        loss = (
            loss_physics_norm
            + WALL_LOSS_WEIGHT * loss_wall_norm
            + CORRECTION_LOSS_WEIGHT * loss_correction
        )

        loss.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step(loss)

        last_loss_physics = float(loss_physics.item())
        last_loss_wall = float(loss_wall.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if epoch % 100 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3e}",
                    "phys": f"{loss_physics.item():.3e}",
                    "wall": f"{loss_wall.item():.3e}",
                }
            )
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                "[steady] epoch %d/%d | total=%.3e phys=%.3e wall=%.3e corr=%.3e | phys_n=%.3e wall_n=%.3e",
                epoch + 1,
                epochs,
                loss.item(),
                loss_physics.item(),
                loss_wall.item(),
                loss_correction.item(),
                loss_physics_norm.item(),
                loss_wall_norm.item(),
            )

    logger.info(
        "Steady PINN training complete. Best loss: %.3e | final phys=%.3e wall=%.3e",
        best_loss,
        last_loss_physics,
        last_loss_wall,
    )
    return model


def train_unsteady_pinn_correction(
    x_int,
    y_int_baseline,
    x_wall,
    y_wall_baseline,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    nt=NT,
):
    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_int = x_int.to(DEVICE)
    y_int_baseline = y_int_baseline.to(DEVICE)
    x_wall = x_wall.to(DEVICE)
    y_wall_baseline = y_wall_baseline.to(DEVICE)

    t_vals = torch.linspace(0, T_END, nt, device=DEVICE)
    logger.info("Training unsteady correction PINN: %d epochs, %d time steps", epochs, nt)
    pbar = tqdm(range(epochs), desc="PINN Correction Training (unsteady)")
    best_total = float("inf")
    last_loss_physics = 0.0
    last_loss_wall = 0.0
    physics_ref = None
    vel_scale = y_int_baseline[:, 0:3].abs().max().detach().clamp_min(1e-8)
    wall_point_factor = max(1.0, float(x_int.shape[0]) / max(1.0, float(x_wall.shape[0])))

    for epoch in pbar:
        optimizer.zero_grad()
        loss_physics = 0.0
        loss_wall = 0.0
        loss_correction = 0.0

        for t in t_vals:
            t_col = t * torch.ones((x_int.shape[0], 1), device=DEVICE)
            xt_int = torch.cat([x_int, t_col], dim=1).requires_grad_(True)
            scale = pulsatile_scale(t)
            delta = model(xt_int)

            du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]
            u = scale * y_int_baseline[:, 0:1] + du
            v = scale * y_int_baseline[:, 1:2] + dv
            w = scale * y_int_baseline[:, 2:3] + dw
            p = y_int_baseline[:, 3:4] + dp

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

            t_col_wall = t * torch.ones((x_wall.shape[0], 1), device=DEVICE)
            xt_wall = torch.cat([x_wall, t_col_wall], dim=1)
            delta_wall = model(xt_wall)
            wall_target = -scale * y_wall_baseline[:, 0:3] / vel_scale
            wall_pred = delta_wall[:, 0:3] / vel_scale
            loss_wall += (wall_pred - wall_target).pow(2).mean() * wall_point_factor

            loss_correction += (
                du.pow(2).mean() + dv.pow(2).mean() + dw.pow(2).mean() + dp.pow(2).mean()
            )

        loss_physics /= nt
        loss_wall /= nt
        loss_correction /= nt
        if physics_ref is None:
            physics_ref = float(loss_physics.item()) + 1e-12

        loss_physics_norm = loss_physics / physics_ref
        loss_wall_norm = loss_wall
        loss = (
            loss_physics_norm
            + WALL_LOSS_WEIGHT * loss_wall_norm
            + CORRECTION_LOSS_WEIGHT * loss_correction
        )

        loss.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        cur_total = float(loss.item())
        best_total = min(best_total, cur_total)
        last_loss_physics = float(loss_physics.item())
        last_loss_wall = float(loss_wall.item())

        if epoch % 100 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3e}",
                    "phys": f"{loss_physics.item():.3e}",
                    "wall": f"{loss_wall.item():.3e}",
                }
            )
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0 or epoch == epochs - 1:
            logger.info(
                "[unsteady] epoch %d/%d | total=%.3e phys=%.3e wall=%.3e corr=%.3e | phys_n=%.3e wall_n=%.3e",
                epoch + 1,
                epochs,
                loss.item(),
                loss_physics.item(),
                loss_wall.item(),
                loss_correction.item(),
                loss_physics_norm.item(),
                loss_wall_norm.item(),
            )

    logger.info(
        "Unsteady correction PINN training complete. Best loss: %.3e | final phys=%.3e wall=%.3e",
        best_total,
        last_loss_physics,
        last_loss_wall,
    )
    return model


# End-to-end case processing
def process_single_case(
    case_name: str,
    vtp_file: Path,
    x_sup: np.ndarray,
    preds: np.ndarray,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    n_interior: int = DEFAULT_N_INTERIOR,
    n_wall: int = DEFAULT_N_WALL,
    unsteady: bool = True,
):
    start_time = time.time()
    logger.info("Processing PINN correction: %s", case_name)

    xyz = x_sup[0, :, 0:3].astype(np.float32)
    pressure = preds[:, 0:1].astype(np.float32)
    velocity = preds[:, 1:4].astype(np.float32)

    wall_points, wall_normals = extract_wall_points(vtp_file, n_wall)

    x_all = torch.tensor(xyz, dtype=torch.float32, device=DEVICE)
    v_all = torch.tensor(velocity, dtype=torch.float32, device=DEVICE)
    p_all = torch.tensor(pressure, dtype=torch.float32, device=DEVICE)
    y_all = torch.cat([v_all, p_all], dim=1)

    x_wall = torch.tensor(wall_points, dtype=torch.float32, device=DEVICE)
    n_wall_t = torch.tensor(wall_normals, dtype=torch.float32, device=DEVICE)

    n_points = len(xyz)
    n_sup = min(n_points, DEFAULT_N_SUP)
    n_int = min(n_points, n_interior)

    idx_sup = np.random.choice(n_points, n_sup, replace=False)
    idx_int = np.linspace(0, n_points - 1, n_int, dtype=int)

    x_sup_t = x_all[idx_sup]
    y_sup_t = y_all[idx_sup]
    x_int = x_all[idx_int]

    tree = KDTree(xyz)
    _, wall_nn_idx = tree.query(wall_points, k=1)
    y_wall_baseline = torch.tensor(
        np.concatenate([velocity[wall_nn_idx], pressure[wall_nn_idx]], axis=1),
        dtype=torch.float32,
        device=DEVICE,
    )
    y_int_baseline = y_all[idx_int]

    logger.info("Training correction PINN...")
    train_start = time.time()

    if unsteady:
        model = train_unsteady_pinn_correction(
            x_int,
            y_int_baseline,
            x_wall,
            y_wall_baseline,
            epochs=epochs,
            lr=DEFAULT_LR,
            nt=NT,
        )
    else:
        model = train_steady_pinn_correction(
            x_int,
            y_int_baseline,
            x_sup_t,
            y_sup_t,
            x_wall,
            y_wall_baseline,
            epochs=epochs,
            lr=DEFAULT_LR,
        )

    train_time = time.time() - train_start
    logger.info("PINN training time: %.1fs", train_time)

    model.eval()
    with torch.no_grad():
        if unsteady:
            velocity_accum = torch.zeros_like(v_all)
            pressure_accum = torch.zeros_like(p_all)
            t_eval = torch.linspace(0, T_END, steps=10, device=DEVICE)
            for t_val in t_eval:
                scale = pulsatile_scale(t_val)
                t_col = t_val * torch.ones((x_all.shape[0], 1), device=DEVICE)
                xt_all = torch.cat([x_all, t_col], dim=1)
                delta_all = model(xt_all)
                velocity_accum += scale * v_all + delta_all[:, :3]
                pressure_accum += p_all + delta_all[:, 3:4]
            velocity_corrected = (velocity_accum / len(t_eval)).cpu().numpy()
            pressure_corrected = (pressure_accum / len(t_eval)).cpu().numpy()
        else:
            t_col = torch.zeros((x_all.shape[0], 1), device=DEVICE)
            xt_all = torch.cat([x_all, t_col], dim=1)
            delta_all = model(xt_all)
            velocity_corrected = (v_all + delta_all[:, :3]).cpu().numpy()
            pressure_corrected = (p_all + delta_all[:, 3:4]).cpu().numpy()

    if unsteady:
        wss_history = []
        t_vals = np.linspace(0, T_END, NT)
        for t in t_vals:
            scale = pulsatile_scale(torch.tensor(t, device=DEVICE))
            t_col_wall = torch.full((x_wall.shape[0], 1), t, device=DEVICE)

            with torch.enable_grad():
                xt_wall = torch.cat([x_wall, t_col_wall], dim=1).requires_grad_(True)
                delta_wall = model(xt_wall)
                vel_wall = scale * y_wall_baseline[:, 0:3] + delta_wall[:, :3]
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
            delta_wall = model(xt_wall)
            vel_wall = y_wall_baseline[:, 0:3] + delta_wall[:, :3]
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
            scale = pulsatile_scale(torch.tensor(t, device=DEVICE))
            t_col_all = torch.full((x_all.shape[0], 1), t, device=DEVICE)
            t_col_wall = torch.full((x_wall.shape[0], 1), t, device=DEVICE)

            with torch.no_grad():
                xt_all = torch.cat([x_all, t_col_all], dim=1)
                delta_all = model(xt_all)
                vel_t = scale * v_all + delta_all[:, :3]
                pres_t = p_all + delta_all[:, 3:4]

            with torch.enable_grad():
                xt_wall = torch.cat([x_wall, t_col_wall], dim=1).requires_grad_(True)
                delta_wall = model(xt_wall)
                vel_wall = scale * y_wall_baseline[:, 0:3] + delta_wall[:, :3]
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
    if not vtp_files:
        logger.error("No VTP files found in %s", vtp_dir)
        sys.exit(1)

    if LIMIT:
        vtp_files = vtp_files[:LIMIT]

    logger.info("Files: %d | Device: %s", len(vtp_files), DEVICE)
    logger.info(
        "Checkpoint: %s | Points: %d | Flow rate: %.4f",
        CHECKPOINT_DIR,
        N_POINTS,
        FLOW_RATE,
    )
    logger.info("=" * 60)

    try:
        trunk, branch_bc, branch_bp = load_deeponet(CHECKPOINT_DIR, DEVICE, CHECKPOINT_ID)
        logger.info("Models loaded successfully")
    except Exception as exc:
        logger.error("Failed to load DeepONet models: %s", exc)
        sys.exit(1)

    stage1_results = []
    stage1_failed = []
    stage2_results = []
    stage2_failed = []

    for vtp_file in vtp_files:
        case = vtp_file.stem
        logger.info("Processing case: %s", case)

        try:
            t0 = time.time()

            t_pre = time.time()
            x_sup, _, _, simple_inlet = preprocess_vtp(vtp_file, N_POINTS, FLOW_RATE)
            preprocess_time = time.time() - t_pre
            logger.info("  Preprocess: %.2fs, %d points", preprocess_time, x_sup.shape[1])

            t_inf = time.time()
            preds = run_inference(x_sup, simple_inlet, trunk, branch_bc, branch_bp, DEVICE)
            inference_time = time.time() - t_inf
            logger.info("  Inference: %.4fs", inference_time)

            points = x_sup[0, :, :3]
            csv_path = Path(BATCH_OUTPUT_DIR) / f"{case}.csv"
            save_csv(points, preds, csv_path)

            if SAVE_NPY:
                save_npy(points, preds, Path(BATCH_OUTPUT_DIR) / f"{case}.npy")
            if SAVE_VTP:
                save_vtp(points, preds, Path(BATCH_OUTPUT_DIR) / f"{case}_predicted.vtp")

            total_stage1 = time.time() - t0
            stage1_results.append(
                {
                    "case_name": case,
                    "n_points": x_sup.shape[1],
                    "preprocess_time": preprocess_time,
                    "inference_time": inference_time,
                    "total_time": total_stage1,
                    "output_csv": str(csv_path),
                }
            )
            logger.info("  Baseline saved: %s (%.2fs)", csv_path.name, total_stage1)
        except Exception as exc:
            logger.error("DeepONet failed for %s: %s", case, exc)
            stage1_failed.append({"file": str(vtp_file), "error": str(exc)})
            continue

        try:
            pinn_result = process_single_case(
                case,
                vtp_file,
                x_sup,
                preds,
                Path(PINN_OUTPUT_DIR),
                epochs=UNSTEADY_EPOCHS if UNSTEADY else STEADY_EPOCHS,
                n_interior=DEFAULT_N_INTERIOR,
                n_wall=DEFAULT_N_WALL,
                unsteady=UNSTEADY,
            )
            stage2_results.append(pinn_result)
        except Exception as exc:
            logger.error("PINN correction failed for %s: %s", case, exc)
            stage2_failed.append({"case": case, "error": str(exc)})

    logger.info("=" * 60)
    logger.info(
        "DeepONet summary: successful %d/%d | failed %d",
        len(stage1_results),
        len(vtp_files),
        len(stage1_failed),
    )
    logger.info(
        "PINN summary: successful %d/%d | failed %d",
        len(stage2_results),
        len(stage1_results),
        len(stage2_failed),
    )

    if stage1_results:
        avg_t = np.mean([r["total_time"] for r in stage1_results])
        avg_i = np.mean([r["inference_time"] for r in stage1_results])
        logger.info("DeepONet avg total: %.2fs | avg inference: %.4fs", avg_t, avg_i)

    if stage1_results:
        stage1_summary = Path(BATCH_OUTPUT_DIR) / "processing_summary.csv"
        with open(stage1_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=stage1_results[0].keys())
            w.writeheader()
            w.writerows(stage1_results)
        logger.info("DeepONet summary saved to %s", stage1_summary)

    if stage1_failed:
        stage1_failed_path = Path(BATCH_OUTPUT_DIR) / "failed_files.csv"
        with open(stage1_failed_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "error"])
            w.writeheader()
            w.writerows(stage1_failed)
        logger.info("DeepONet failures saved to %s", stage1_failed_path)

    if stage2_results:
        stage2_summary = Path(PINN_OUTPUT_DIR) / "processing_summary.csv"
        with open(stage2_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=stage2_results[0].keys())
            w.writeheader()
            w.writerows(stage2_results)
        logger.info("PINN summary saved to %s", stage2_summary)

    if stage2_failed:
        stage2_failed_path = Path(PINN_OUTPUT_DIR) / "failed_cases.csv"
        with open(stage2_failed_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "error"])
            w.writeheader()
            w.writerows(stage2_failed)
        logger.info("PINN failures saved to %s", stage2_failed_path)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
