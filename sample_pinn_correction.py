#!/usr/bin/env python3
# Version 3 source snapshot
"""
sample_pinn_correction.py

Run a single-case PINN correction (steady or unsteady) and save per-epoch
losses/residuals to a CSV for inspection.

Example:
  python sample_pinn_correction.py --csv-file predictions/batch_results/C0001_cut1.csv \
      --vtp-file vtp_data/C0001_cut1.vtp --output-dir samples --epochs 500 --steady

This script re-uses model architectures from `pinn_correction_batch.py`.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pinn_correction_batch as pcb
import torch
from tqdm import tqdm


def train_and_log_steady(
    X_int,
    Y_int_baseline,
    X_sup,
    Y_sup_baseline,
    X_wall,
    Y_wall_baseline,
    output_csv: Path,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    model = pcb.CorrectionPINN(hidden_dim=128, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_int = X_int.to(device)
    Y_int_baseline = Y_int_baseline.to(device)
    X_sup = X_sup.to(device)
    Y_sup_baseline = Y_sup_baseline.to(device)
    X_wall = X_wall.to(device)
    Y_wall_baseline = Y_wall_baseline.to(device)

    # CSV header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "loss",
                "loss_physics",
                "loss_wall",
                "loss_correction",
                "mom_u_mean",
                "mom_v_mean",
                "mom_w_mean",
                "cont_mean",
            ]
        )

    best_loss = float("inf")
    pbar = tqdm(range(epochs), desc="Steady PINN")

    for epoch in pbar:
        optimizer.zero_grad()

        # Build input with dummy time=0
        t_zeros_int = torch.zeros((X_int.shape[0], 1), device=device)
        XT_int = torch.cat([X_int, t_zeros_int], dim=1).requires_grad_(True)

        delta = model(XT_int)
        du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]

        # Corrected field
        u = Y_int_baseline[:, 0:1] + du
        v = Y_int_baseline[:, 1:2] + dv
        w = Y_int_baseline[:, 2:3] + dw
        p = Y_int_baseline[:, 3:4] + dp

        # Derivatives
        u_x = pcb.gradients(u, XT_int)[:, 0:1]
        u_y = pcb.gradients(u, XT_int)[:, 1:2]
        u_z = pcb.gradients(u, XT_int)[:, 2:3]

        v_x = pcb.gradients(v, XT_int)[:, 0:1]
        v_y = pcb.gradients(v, XT_int)[:, 1:2]
        v_z = pcb.gradients(v, XT_int)[:, 2:3]

        w_x = pcb.gradients(w, XT_int)[:, 0:1]
        w_y = pcb.gradients(w, XT_int)[:, 1:2]
        w_z = pcb.gradients(w, XT_int)[:, 2:3]

        p_x = pcb.gradients(p, XT_int)[:, 0:1]
        p_y = pcb.gradients(p, XT_int)[:, 1:2]
        p_z = pcb.gradients(p, XT_int)[:, 2:3]

        # Laplacians
        u_xx = pcb.gradients(u_x, XT_int)[:, 0:1]
        u_yy = pcb.gradients(u_y, XT_int)[:, 1:2]
        u_zz = pcb.gradients(u_z, XT_int)[:, 2:3]

        v_xx = pcb.gradients(v_x, XT_int)[:, 0:1]
        v_yy = pcb.gradients(v_y, XT_int)[:, 1:2]
        v_zz = pcb.gradients(v_z, XT_int)[:, 2:3]

        w_xx = pcb.gradients(w_x, XT_int)[:, 0:1]
        w_yy = pcb.gradients(w_y, XT_int)[:, 1:2]
        w_zz = pcb.gradients(w_z, XT_int)[:, 2:3]

        # Residuals
        mom_u = (u * u_x + v * u_y + w * u_z) + p_x - pcb.NU * (u_xx + u_yy + u_zz)
        mom_v = (u * v_x + v * v_y + w * v_z) + p_y - pcb.NU * (v_xx + v_yy + v_zz)
        mom_w = (u * w_x + v * w_y + w * w_z) + p_z - pcb.NU * (w_xx + w_yy + w_zz)
        cont = u_x + v_y + w_z

        loss_physics = (
            mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
        )

        # Wall loss
        t_zeros_wall = torch.zeros((X_wall.shape[0], 1), device=device)
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

        # Logging
        mom_u_mean = mom_u.abs().mean().item()
        mom_v_mean = mom_v.abs().mean().item()
        mom_w_mean = mom_w.abs().mean().item()
        cont_mean = cont.abs().mean().item()

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    loss.item(),
                    loss_physics.item(),
                    loss_wall.item(),
                    loss_correction.item(),
                    mom_u_mean,
                    mom_v_mean,
                    mom_w_mean,
                    cont_mean,
                ]
            )

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.3e}"})

    return model


def train_and_log_unsteady(
    X_int,
    Y_int_baseline,
    X_wall,
    Y_wall_baseline,
    nt: int,
    output_csv: Path,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    model = pcb.CorrectionPINN(hidden_dim=128, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_int = X_int.to(device)
    Y_int_baseline = Y_int_baseline.to(device)
    X_wall = X_wall.to(device)
    Y_wall_baseline = Y_wall_baseline.to(device)

    t_vals = torch.linspace(0, pcb.T_END, nt, device=device)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "loss",
                "loss_physics",
                "loss_wall",
                "loss_correction",
                "mom_u_mean",
                "mom_v_mean",
                "mom_w_mean",
                "cont_mean",
            ]
        )

    pbar = tqdm(range(epochs), desc="Unsteady PINN")
    best_loss = float("inf")

    for epoch in pbar:
        optimizer.zero_grad()
        loss_physics = 0.0
        loss_wall = 0.0
        loss_correction = 0.0
        mom_u_acc = 0.0
        mom_v_acc = 0.0
        mom_w_acc = 0.0
        cont_acc = 0.0

        for t in t_vals:
            t_col = t * torch.ones((X_int.shape[0], 1), device=device)
            XT_int = torch.cat([X_int, t_col], dim=1).requires_grad_(True)

            scale = pcb.pulsatile_scale(t)
            delta = model(XT_int)
            du, dv, dw, dp = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], delta[:, 3:4]

            u = scale * Y_int_baseline[:, 0:1] + du
            v = scale * Y_int_baseline[:, 1:2] + dv
            w = scale * Y_int_baseline[:, 2:3] + dw
            p = Y_int_baseline[:, 3:4] + dp

            # time and spatial derivatives
            u_t = pcb.gradients(u, XT_int)[:, 3:4]
            v_t = pcb.gradients(v, XT_int)[:, 3:4]
            w_t = pcb.gradients(w, XT_int)[:, 3:4]

            u_x = pcb.gradients(u, XT_int)[:, 0:1]
            u_y = pcb.gradients(u, XT_int)[:, 1:2]
            u_z = pcb.gradients(u, XT_int)[:, 2:3]

            v_x = pcb.gradients(v, XT_int)[:, 0:1]
            v_y = pcb.gradients(v, XT_int)[:, 1:2]
            v_z = pcb.gradients(v, XT_int)[:, 2:3]

            w_x = pcb.gradients(w, XT_int)[:, 0:1]
            w_y = pcb.gradients(w, XT_int)[:, 1:2]
            w_z = pcb.gradients(w, XT_int)[:, 2:3]

            p_x = pcb.gradients(p, XT_int)[:, 0:1]
            p_y = pcb.gradients(p, XT_int)[:, 1:2]
            p_z = pcb.gradients(p, XT_int)[:, 2:3]

            u_xx = pcb.gradients(u_x, XT_int)[:, 0:1]
            u_yy = pcb.gradients(u_y, XT_int)[:, 1:2]
            u_zz = pcb.gradients(u_z, XT_int)[:, 2:3]

            v_xx = pcb.gradients(v_x, XT_int)[:, 0:1]
            v_yy = pcb.gradients(v_y, XT_int)[:, 1:2]
            v_zz = pcb.gradients(v_z, XT_int)[:, 2:3]

            w_xx = pcb.gradients(w_x, XT_int)[:, 0:1]
            w_yy = pcb.gradients(w_y, XT_int)[:, 1:2]
            w_zz = pcb.gradients(w_z, XT_int)[:, 2:3]

            mom_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - pcb.NU * (u_xx + u_yy + u_zz)
            mom_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - pcb.NU * (v_xx + v_yy + v_zz)
            mom_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - pcb.NU * (w_xx + w_yy + w_zz)
            cont = u_x + v_y + w_z

            loss_physics += (
                mom_u.pow(2).mean() + mom_v.pow(2).mean() + mom_w.pow(2).mean() + cont.pow(2).mean()
            )

            # wall BC
            t_col_wall = t * torch.ones((X_wall.shape[0], 1), device=device)
            XT_wall = torch.cat([X_wall, t_col_wall], dim=1)
            delta_wall = model(XT_wall)
            u_wall = scale * Y_wall_baseline[:, 0:1] + delta_wall[:, 0:1]
            v_wall = scale * Y_wall_baseline[:, 1:2] + delta_wall[:, 1:2]
            w_wall = scale * Y_wall_baseline[:, 2:3] + delta_wall[:, 2:3]
            loss_wall += u_wall.pow(2).mean() + v_wall.pow(2).mean() + w_wall.pow(2).mean()

            loss_correction += (
                du.pow(2).mean() + dv.pow(2).mean() + dw.pow(2).mean() + dp.pow(2).mean()
            )

            mom_u_acc += mom_u.abs().mean().item()
            mom_v_acc += mom_v.abs().mean().item()
            mom_w_acc += mom_w.abs().mean().item()
            cont_acc += cont.abs().mean().item()

        # Average over time
        loss_physics = loss_physics / nt
        loss_wall = loss_wall / nt
        loss_correction = loss_correction / nt
        mom_u_mean = mom_u_acc / nt
        mom_v_mean = mom_v_acc / nt
        mom_w_mean = mom_w_acc / nt
        cont_mean = cont_acc / nt

        loss = loss_physics + 2.0 * loss_wall + 0.1 * loss_correction
        loss.backward()
        optimizer.step()

        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    loss.item(),
                    loss_physics.item(),
                    loss_wall.item(),
                    loss_correction.item(),
                    mom_u_mean,
                    mom_v_mean,
                    mom_w_mean,
                    cont_mean,
                ]
            )

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.3e}"})

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Sample PINN correction for one case with CSV logging"
    )
    parser.add_argument(
        "--csv-file", type=str, required=True, help="DeepONet prediction CSV for a single case"
    )
    parser.add_argument(
        "--vtp-file", type=str, required=True, help="Corresponding VTP geometry file"
    )
    parser.add_argument("--output-dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--n-interior", type=int, default=2048, help="Interior points")
    parser.add_argument("--n-wall", type=int, default=1024, help="Wall points")
    parser.add_argument("--nt", type=int, default=8, help="Time samples for unsteady")
    parser.add_argument(
        "--steady", action="store_true", help="Run steady PINN (default is unsteady)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    csv_file = Path(args.csv_file)
    vtp_file = Path(args.vtp_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline
    xyz, velocity, pressure = pcb.load_deeponet_predictions(csv_file)
    wall_points, wall_normals = pcb.extract_wall_points(vtp_file, n_wall_points=args.n_wall)

    X_all = torch.tensor(xyz, dtype=torch.float32)
    V_all = torch.tensor(velocity, dtype=torch.float32)
    P_all = torch.tensor(pressure, dtype=torch.float32)
    Y_all = torch.cat([V_all, P_all], dim=1)

    # sample points
    n_points = len(xyz)
    n_int = min(n_points, args.n_interior)
    n_sup = min(n_points, max(128, args.n_interior // 4))
    idx_int = np.random.choice(n_points, n_int, replace=False)
    idx_sup = np.random.choice(n_points, n_sup, replace=False)

    X_int = X_all[idx_int]
    Y_int_baseline = Y_all[idx_int]
    X_sup = X_all[idx_sup]
    Y_sup = Y_all[idx_sup]

    # baseline at wall points
    from scipy.spatial import KDTree

    tree = KDTree(xyz)
    _, wall_nn_idx = tree.query(wall_points, k=1)
    Y_wall_baseline = torch.tensor(
        np.concatenate([velocity[wall_nn_idx], pressure[wall_nn_idx]], axis=1), dtype=torch.float32
    )
    X_wall = torch.tensor(wall_points, dtype=torch.float32)

    device = torch.device(args.device)

    # prepare outputs
    case_name = csv_file.stem.replace("_corrected", "")
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    losses_csv = case_dir / "losses_per_epoch.csv"

    start = time.time()
    if args.steady:
        model = train_and_log_steady(
            X_int,
            Y_int_baseline,
            X_sup,
            Y_sup,
            X_wall,
            Y_wall_baseline,
            output_csv=losses_csv,
            epochs=args.epochs,
            lr=1e-3,
            device=device,
        )
    else:
        model = train_and_log_unsteady(
            X_int,
            Y_int_baseline,
            X_wall,
            Y_wall_baseline,
            nt=args.nt,
            output_csv=losses_csv,
            epochs=args.epochs,
            lr=1e-3,
            device=device,
        )

    elapsed = time.time() - start
    # save model
    model_file = case_dir / "sample_correction_model.pt"
    torch.save(model.state_dict(), model_file)

    print(f"Completed sample correction for {case_name} in {elapsed:.1f}s")
    print(f"Losses saved to: {losses_csv}")
    print(f"Model saved to: {model_file}")


if __name__ == "__main__":
    main()
