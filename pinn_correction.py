# Version 4 source snapshot
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy.spatial import KDTree
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pyvista as pv

HAS_PYVISTA = True

# Configuration
PREDICTIONS_DIR = "predictions/batch_results"
VTP_DIR = "vtp_data"
OUTPUT_DIR = "predictions/optimized_pinn"
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
LIMIT = None
MU = 0.0035
RHO = 1060.0
NU = MU / RHO
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
log_dir = Path(OUTPUT_DIR)
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("PINNCorrection")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(fmt)
fh = logging.FileHandler(log_dir / "pinn_correction.log")
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)


# Utility: gradients
def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


# Neural net for correction
class CorrectionPINN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Pulsatile scaling
def pulsatile_scale(t: torch.Tensor, t_end: float = T_END) -> torch.Tensor:
    return 1.0 + 0.6 * torch.sin(2 * np.pi * t / t_end)


# I/O and geometry helpers
def load_deeponet_predictions(csv_file: Path):
    data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
    xyz = data[:, 0:3].astype(np.float32)
    pressure = data[:, 3:4].astype(np.float32)
    velocity = data[:, 4:7].astype(np.float32)
    return xyz, velocity, pressure


def extract_wall_points(vtp_file: Path, n_wall_points: int = DEFAULT_N_WALL):
    mesh = pv.read(vtp_file) if HAS_PYVISTA else None
    if mesh is None:
        # fallback: empty arrays
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if isinstance(mesh, pv.UnstructuredGrid):
        surface = mesh.extract_surface()
    else:
        surface = mesh

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
    velocity: torch.Tensor,
    points: torch.Tensor,
    wall_points: torch.Tensor,
    wall_normals: torch.Tensor,
    mu: float = MU,
) -> torch.Tensor:
    tree = KDTree(points.detach().cpu().numpy())
    distances, indices = tree.query(wall_points.detach().cpu().numpy(), k=5)
    indices_tensor = torch.tensor(indices, device=velocity.device)
    nearby_vels = velocity[indices_tensor]
    distances_tensor = torch.tensor(distances, device=velocity.device, dtype=torch.float32)
    distances_tensor = distances_tensor.clamp(min=1e-6)
    avg_vel = nearby_vels.mean(dim=1)
    avg_dist = distances_tensor.mean(dim=1, keepdim=True)
    wall_normals_unit = wall_normals / (torch.norm(wall_normals, dim=-1, keepdim=True) + 1e-8)
    vel_normal = torch.sum(avg_vel * wall_normals_unit, dim=-1, keepdim=True) * wall_normals_unit
    vel_tangent = avg_vel - vel_normal
    wss_vector = mu * vel_tangent / avg_dist
    return wss_vector


def compute_wss_magnitude(wss_vector: torch.Tensor) -> torch.Tensor:
    return torch.norm(wss_vector, dim=-1)


def compute_osi(wss_history):
    wss_stack = torch.stack(wss_history, dim=0)
    wss_avg = wss_stack.mean(dim=0)
    wss_mag_avg = torch.norm(wss_stack, dim=-1).mean(dim=0)
    wss_avg_mag = torch.norm(wss_avg, dim=-1)
    osi = 0.5 * (1.0 - wss_avg_mag / (wss_mag_avg + 1e-8))
    return osi


def compute_von_mises_stress(wss_vector: torch.Tensor):
    wss_mag = torch.norm(wss_vector, dim=-1)
    return np.sqrt(3) * wss_mag


def compute_tawss(wss_history):
    wss_mags = torch.stack([torch.norm(wss, dim=-1) for wss in wss_history], dim=0)
    return wss_mags.mean(dim=0)


# Training functions (steady & unsteady correction)
def train_steady_pinn_correction(
    X_int,
    Y_int_baseline,
    X_sup,
    Y_sup_baseline,
    X_wall,
    Y_wall_baseline,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
):
    log = logger.info
    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200
    )

    X_int = X_int.to(DEVICE).requires_grad_(True)
    Y_int_baseline = Y_int_baseline.to(DEVICE)
    X_sup = X_sup.to(DEVICE)
    Y_sup_baseline = Y_sup_baseline.to(DEVICE)
    X_wall = X_wall.to(DEVICE)
    Y_wall_baseline = Y_wall_baseline.to(DEVICE)

    log(
        f"Training steady correction PINN: {epochs} epochs, {X_int.shape[0]} interior pts, {X_wall.shape[0]} wall pts"
    )
    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="PINN Correction Training")

    for epoch in pbar:
        optimizer.zero_grad()
        t_zeros_int = torch.zeros((X_int.shape[0], 1), device=DEVICE)
        XT_int = torch.cat([X_int, t_zeros_int], dim=1).requires_grad_(True)
        delta = model(XT_int)
        du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]
        u = Y_int_baseline[:, 0:1] + du
        v = Y_int_baseline[:, 1:2] + dv
        w = Y_int_baseline[:, 2:3] + dw
        p = Y_int_baseline[:, 3:4] + dp

        # derivatives and residuals
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

        mom_u = (u * u_x + v * u_y + w * u_z) + p_x - NU * (u_xx + u_yy + u_zz)
        mom_v = (u * v_x + v * v_y + w * v_z) + p_y - NU * (v_xx + v_yy + v_zz)
        mom_w = (u * w_x + v * w_y + w * w_z) + p_z - NU * (w_xx + w_yy + w_zz)
        cont = u_x + v_y + w_z

        loss_physics = (
            mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
        )

        t_zeros_wall = torch.zeros((X_wall.shape[0], 1), device=DEVICE)
        XT_wall = torch.cat([X_wall, t_zeros_wall], dim=1)
        delta_wall = model(XT_wall)
        u_wall = Y_wall_baseline[:, 0:1] + delta_wall[:, 0:1]
        v_wall = Y_wall_baseline[:, 1:2] + delta_wall[:, 1:2]
        w_wall = Y_wall_baseline[:, 2:3] + delta_wall[:, 2:3]
        loss_wall = u_wall.pow(2).mean() + v_wall.pow(2).mean() + w_wall.pow(2).mean()

        loss_correction = du.pow(2).mean() + dv.pow(2).mean() + dw.pow(2).mean() + dp.pow(2).mean()
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
                }
            )

    log(f"Training complete. Best loss: {best_loss:.3e}")
    return model


def train_unsteady_pinn_correction(
    X_int, Y_int_baseline, X_wall, Y_wall_baseline, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, nt=NT
):
    log = logger.info
    model = CorrectionPINN(hidden_dim=128, num_layers=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_int = X_int.to(DEVICE)
    Y_int_baseline = Y_int_baseline.to(DEVICE)
    X_wall = X_wall.to(DEVICE)
    Y_wall_baseline = Y_wall_baseline.to(DEVICE)

    t_vals = torch.linspace(0, T_END, nt, device=DEVICE)
    log(f"Training unsteady correction PINN: {epochs} epochs, {nt} time steps")
    pbar = tqdm(range(epochs), desc="PINN Correction Training")

    for epoch in pbar:
        optimizer.zero_grad()
        loss_physics = 0.0
        loss_wall = 0.0
        loss_correction = 0.0

        for t in t_vals:
            t_col = t * torch.ones((X_int.shape[0], 1), device=DEVICE)
            XT_int = torch.cat([X_int, t_col], dim=1).requires_grad_(True)
            scale = pulsatile_scale(t)
            delta = model(XT_int)
            du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]
            u = scale * Y_int_baseline[:, 0:1] + du
            v = scale * Y_int_baseline[:, 1:2] + dv
            w = scale * Y_int_baseline[:, 2:3] + dw
            p = Y_int_baseline[:, 3:4] + dp

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

            mom_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - NU * (u_xx + u_yy + u_zz)
            mom_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - NU * (v_xx + v_yy + v_zz)
            mom_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - NU * (w_xx + w_yy + w_zz)
            cont = u_x + v_y + w_z

            loss_physics += (
                mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
            )

            t_col_wall = t * torch.ones((X_wall.shape[0], 1), device=DEVICE)
            XT_wall = torch.cat([X_wall, t_col_wall], dim=1)
            delta_wall = model(XT_wall)
            u_wall = scale * Y_wall_baseline[:, 0:1] + delta_wall[:, 0:1]
            v_wall = scale * Y_wall_baseline[:, 1:2] + delta_wall[:, 1:2]
            w_wall = scale * Y_wall_baseline[:, 2:3] + delta_wall[:, 2:3]
            loss_wall += u_wall.pow(2).mean() + v_wall.pow(2).mean() + w_wall.pow(2).mean()

            loss_correction += (
                du.pow(2).mean() + dv.pow(2).mean() + dw.pow(2).mean() + dp.pow(2).mean()
            )

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


def process_single_case(
    csv_file: Path,
    vtp_file: Path,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    n_interior: int = DEFAULT_N_INTERIOR,
    n_wall: int = DEFAULT_N_WALL,
    unsteady: bool = True,
):
    log = logger.info
    case_name = csv_file.stem
    start_time = time.time()
    log(f"Processing: {case_name}")

    xyz, velocity, pressure = load_deeponet_predictions(csv_file)
    wall_points, wall_normals = extract_wall_points(vtp_file, n_wall)

    X_all = torch.tensor(xyz, dtype=torch.float32, device=DEVICE)
    V_all = torch.tensor(velocity, dtype=torch.float32, device=DEVICE)
    P_all = torch.tensor(pressure, dtype=torch.float32, device=DEVICE)
    Y_all = torch.cat([V_all, P_all], dim=1)

    X_wall = torch.tensor(wall_points, dtype=torch.float32, device=DEVICE)
    N_wall = torch.tensor(wall_normals, dtype=torch.float32, device=DEVICE)

    n_points = len(xyz)
    n_sup = min(n_points, DEFAULT_N_SUP)
    n_int = min(n_points, n_interior)
    idx_sup = np.random.choice(n_points, n_sup, replace=False)
    idx_int = np.random.choice(n_points, n_int, replace=False)
    X_sup = X_all[idx_sup]
    Y_sup = Y_all[idx_sup]
    X_int = X_all[idx_int]

    tree = KDTree(xyz)
    _, wall_nn_idx = tree.query(wall_points, k=1)
    Y_wall_baseline = torch.tensor(
        np.concatenate([velocity[wall_nn_idx], pressure[wall_nn_idx]], axis=1),
        dtype=torch.float32,
        device=DEVICE,
    )
    Y_int_baseline = Y_all[idx_int]

    log("Training correction PINN...")
    train_start = time.time()
    if unsteady:
        model = train_unsteady_pinn_correction(
            X_int, Y_int_baseline, X_wall, Y_wall_baseline, epochs=epochs, lr=DEFAULT_LR, nt=NT
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
            lr=DEFAULT_LR,
        )
    train_time = time.time() - train_start
    log(f"PINN training time: {train_time:.1f}s")

    model.eval()
    with torch.no_grad():
        if unsteady:
            t_mid = T_END / 2
            scale = pulsatile_scale(torch.tensor(t_mid, device=DEVICE))
            t_col = t_mid * torch.ones((X_all.shape[0], 1), device=DEVICE)
            XT_all = torch.cat([X_all, t_col], dim=1)
            delta_all = model(XT_all)
            velocity_corrected = (scale * V_all + delta_all[:, :3]).cpu().numpy()
            pressure_corrected = (P_all + delta_all[:, 3:4]).cpu().numpy()
        else:
            t_col = torch.zeros((X_all.shape[0], 1), device=DEVICE)
            XT_all = torch.cat([X_all, t_col], dim=1)
            delta_all = model(XT_all)
            velocity_corrected = (V_all + delta_all[:, :3]).cpu().numpy()
            pressure_corrected = (P_all + delta_all[:, 3:4]).cpu().numpy()

    if unsteady:
        wss_history = []
        t_vals = np.linspace(0, T_END, NT)
        for t in t_vals:
            scale = pulsatile_scale(torch.tensor(t, device=DEVICE))
            t_col = torch.full((X_all.shape[0], 1), t, device=DEVICE)
            XT = torch.cat([X_all, t_col], dim=1)
            with torch.no_grad():
                delta = model(XT)
                vel = scale * V_all + delta[:, :3]
            wss = compute_wss(vel, X_all, X_wall, N_wall)
            wss_history.append(wss)
        wss_final = wss_history[NT // 2]
        tawss = compute_tawss(wss_history)
        osi = compute_osi(wss_history)
        von_mises = compute_von_mises_stress(wss_final)
    else:
        t_col = torch.zeros((X_all.shape[0], 1), device=DEVICE)
        XT = torch.cat([X_all, t_col], dim=1)
        with torch.no_grad():
            delta = model(XT)
            vel = V_all + delta[:, :3]
        wss_final = compute_wss(vel, X_all, X_wall, N_wall)
        wss_mag = compute_wss_magnitude(wss_final)
        von_mises = compute_von_mises_stress(wss_final)
        tawss = wss_mag
        osi = torch.zeros_like(wss_mag)

    # Save results
    case_dir = Path(output_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    timesteps_dir = case_dir / "timesteps"
    timesteps_dir.mkdir(parents=True, exist_ok=True)

    if unsteady:
        t_vals = np.linspace(0, T_END, NT)
        for ti, t in enumerate(t_vals):
            scale = pulsatile_scale(torch.tensor(t, device=DEVICE))
            t_col = torch.full((X_all.shape[0], 1), t, device=DEVICE)
            XT = torch.cat([X_all, t_col], dim=1)
            with torch.no_grad():
                delta = model(XT)
                vel_t = scale * V_all + delta[:, :3]
                pres_t = P_all + delta[:, 3:4]
            wss_t = compute_wss(vel_t, X_all, X_wall, N_wall)
            wss_mag_t = compute_wss_magnitude(wss_t)
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
            [wall_points, wss_final.cpu().numpy(), wss_mag.cpu().numpy().reshape(-1, 1)], axis=1
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
    log(f"Completed in {total_time:.1f}s")
    return {
        "case_name": case_name,
        "n_points": n_points,
        "n_wall": len(wall_points),
        "train_time": train_time,
        "total_time": total_time,
        "output_dir": str(case_dir),
    }


# Batch pipeline
logger.info(f"Device: {DEVICE}  |  NT: {NT}  |  Unsteady epochs: {UNSTEADY_EPOCHS}")

pred_dir = Path(PREDICTIONS_DIR)
vtp_dir = Path(VTP_DIR)
pairs = []
for csv_file in sorted(pred_dir.glob("*.csv")):
    if csv_file.name.startswith("processing_summary") or csv_file.name.startswith("failed"):
        continue
    base = csv_file.stem.replace("_flow", "").replace("_corrected", "")
    for suffix in [".vtp", "_cut1.vtp"]:
        candidate = vtp_dir / f"{base}{suffix}"
        if candidate.exists():
            pairs.append((csv_file, candidate))
            break

if LIMIT:
    pairs = pairs[:LIMIT]

if not pairs:
    logger.error(f"No matching CSV/VTP pairs in {PREDICTIONS_DIR} + {VTP_DIR}")
    sys.exit(1)

logger.info(f"Found {len(pairs)} pairs")
results = []
failed = []
for csv_path, vtp_path in pairs:
    try:
        results.append(
            process_single_case(
                csv_path,
                vtp_path,
                OUTPUT_DIR,
                epochs=UNSTEADY_EPOCHS,
                n_interior=DEFAULT_N_INTERIOR,
                n_wall=DEFAULT_N_WALL,
                unsteady=True,
            )
        )
    except Exception as e:
        logger.error(f"Failed {csv_path.stem}: {e}")
        import traceback

        traceback.print_exc()
        failed.append({"case": csv_path.stem, "error": str(e)})

logger.info("=" * 60)
logger.info(f"Successful: {len(results)}/{len(pairs)}   Failed: {len(failed)}")
if results:
    summary = Path(OUTPUT_DIR) / "processing_summary.csv"
    with open(summary, "w", newline="") as f:
        import csv as _csv

        w = _csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    logger.info(f"Summary saved to {summary}")

if failed:
    fp = Path(OUTPUT_DIR) / "failed_cases.csv"
    with open(fp, "w", newline="") as f:
        import csv as _csv

        w = _csv.DictWriter(f, fieldnames=["case", "error"])
        w.writeheader()
        w.writerows(failed)
    logger.info(f"Failed logged to {fp}")
