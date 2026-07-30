# Version 3 source snapshot
import argparse
import glob
import os

import numpy as np
import pandas as pd


def find_normals_in_df(df):
    """Try common normal column name sets and return Nx3 array or None."""
    candidates = [
        ("nx", "ny", "nz"),
        ("normal_x", "normal_y", "normal_z"),
        ("nx_local", "ny_local", "nz_local"),
    ]
    for cx, cy, cz in candidates:
        if {cx, cy, cz}.issubset(df.columns):
            return df[[cx, cy, cz]].values
    return None


def load_timesteps_matched(timestep_dir, agg_coords, mu, delta_n, normals=None):
    """
    Load timesteps by matching coordinates between aggregate points and
    timestep files. Prefers `wss_t*.csv` for shear vector; falls back to
    `flow_t*.csv` velocities when needed.

    Args:
        timestep_dir: path to timesteps folder
        agg_coords: (N,3) array of aggregate x,y,z coordinates to match
        mu, delta_n: viscosity and wall-normal distance
        normals: optional (N,3) normals to compute tangential velocity

    Returns:
        p_all: (T, N)
        tau_all: (T, N, 3)
    """
    # Use nearest-neighbor matching between aggregate coords and timestep files
    import numpy as _np

    agg_np = _np.asarray(agg_coords, dtype=float)
    N = agg_np.shape[0]

    flow_files = sorted(glob.glob(os.path.join(timestep_dir, "flow_t*.csv")))
    if len(flow_files) == 0:
        raise RuntimeError("No flow_t* timestep files found in: %s" % timestep_dir)

    p_list = []
    tau_list = []

    for f in flow_files:
        basename = os.path.basename(f)
        idx = basename.split(".")[0].replace("flow_", "")
        wss_path = os.path.join(timestep_dir, f"wss_{idx}.csv")

        df_flow = pd.read_csv(f)
        flow_coords = _np.vstack(
            [df_flow["x"].values, df_flow["y"].values, df_flow["z"].values]
        ).T.astype(float)

        # For each aggregate point find closest flow point index
        mapping = _np.empty(N, dtype=int)
        for i in range(N):
            dif = flow_coords - agg_np[i : i + 1, :]
            d2 = _np.sum(dif * dif, axis=1)
            mapping[i] = int(_np.argmin(d2))

        p_t = _np.empty(N, dtype=float)
        tau_t = _np.empty((N, 3), dtype=float)

        # If wss file exists, use its vectors (map similarly)
        if os.path.exists(wss_path):
            df_wss = pd.read_csv(wss_path)
            wss_coords = _np.vstack(
                [df_wss["x"].values, df_wss["y"].values, df_wss["z"].values]
            ).T.astype(float)
            # for each agg point find closest wss index
            wss_map = _np.empty(N, dtype=int)
            for i in range(N):
                dif = wss_coords - agg_np[i : i + 1, :]
                d2 = _np.sum(dif * dif, axis=1)
                wss_map[i] = int(_np.argmin(d2))

            for out_idx in range(N):
                flow_row = df_flow.iloc[mapping[out_idx]]
                wss_row = df_wss.iloc[wss_map[out_idx]]
                p_t[out_idx] = float(flow_row["p"])
                tau_t[out_idx, 0] = float(wss_row.get("wss_x", 0.0))
                tau_t[out_idx, 1] = float(wss_row.get("wss_y", 0.0))
                tau_t[out_idx, 2] = float(wss_row.get("wss_z", 0.0))
        else:
            for out_idx in range(N):
                flow_row = df_flow.iloc[mapping[out_idx]]
                p_t[out_idx] = float(flow_row["p"])
                vel = _np.array([flow_row["u"], flow_row["v"], flow_row["w"]], dtype=float)
                if normals is not None:
                    n = normals[out_idx]
                    proj = _np.dot(vel, n)
                    vel_tangent = vel - proj * n
                else:
                    vel_tangent = vel
                tau_t[out_idx, :] = mu * vel_tangent / delta_n

        p_list.append(p_t)
        tau_list.append(tau_t)

    p_all = _np.stack(p_list, axis=0)
    tau_all = _np.stack(tau_list, axis=0)
    return p_all, tau_all


def compute_von_mises_equivalent(p, tau):
    """
    Compute RMS-equivalent von Mises wall stress under thin-wall assumption:
        sigma_vm_rms = sqrt( mean(p^2) + 3 * mean(||tau||^2) )
    where means are across time (axis=0).
    """
    p2_mean = np.mean(p**2, axis=0)  # (N,)
    tau2_mean = np.mean(np.sum(tau**2, axis=-1), axis=0)  # (N,)
    return np.sqrt(p2_mean + 3.0 * tau2_mean)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute von Mises-like wall stress from timesteps")
    p.add_argument(
        "--case",
        "-c",
        default="C0001_cut1",
        help="Case directory containing timesteps and aggregate CSV",
    )
    p.add_argument("--mu", type=float, default=0.0035, help="Dynamic viscosity (Pa·s)")
    p.add_argument("--delta-n", type=float, default=1e-4, help="Effective wall-normal distance (m)")
    p.add_argument(
        "--agg-file",
        default=None,
        help="Path to aggregate CSV (overrides case/ hemodynamics_aggregate.csv)",
    )
    p.add_argument(
        "--batch-dir",
        default="predictions/pinn_corrected",
        help="Directory containing case subfolders to process in batch",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mu = args.mu
    delta_n = args.delta_n

    # If batch dir specified and exists, run batch first and exit
    if args.batch_dir:
        batch_dir = args.batch_dir
        if os.path.exists(batch_dir) and os.path.isdir(batch_dir):
            cases = sorted(
                [d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d))]
            )
            if len(cases) == 0:
                raise RuntimeError(f"No subfolders found in batch dir: {batch_dir}")
            print(f"[INFO] Processing batch directory: {batch_dir} -> {len(cases)} cases")
            for c in cases:
                case_path = os.path.join(batch_dir, c)
                try:
                    process_case(case_path, mu, delta_n, None)
                except Exception as e:
                    print(f"[ERROR] Case {case_path} failed: {e}")
            print("[DONE] Batch processing complete.")
            return
        else:
            print(
                f"[WARN] Batch dir not found or not a directory: {args.batch_dir}; falling back to single case."
            )

    # Single-case processing
    case_dir = args.case
    timestep_dir = os.path.join(case_dir, "timesteps")
    agg_file = (
        args.agg_file if args.agg_file else os.path.join(case_dir, "hemodynamics_aggregate.csv")
    )

    print(f"[INFO] Case: {case_dir}")
    print(f"[INFO] Looking for timesteps in: {timestep_dir}")

    if not os.path.exists(agg_file):
        raise RuntimeError(f"Aggregate file not found: {agg_file}")

    process_case(case_dir, mu, delta_n, agg_file)


def process_case(case_dir, mu, delta_n, agg_file_override=None):
    """Process a single case directory: read agg, match timesteps, compute von_mises, save CSV."""
    print(f"[INFO] Processing case: {case_dir}")
    timestep_dir = os.path.join(case_dir, "timesteps")
    agg_file = (
        agg_file_override
        if agg_file_override
        else os.path.join(case_dir, "hemodynamics_aggregate.csv")
    )

    if not os.path.exists(agg_file):
        raise RuntimeError(f"Aggregate file not found: {agg_file}")

    agg = pd.read_csv(agg_file)
    normals = find_normals_in_df(agg)
    if normals is not None:
        print(
            "[INFO] Found wall normals in aggregate CSV; using tangential velocity for WSS proxy."
        )
    else:
        print(
            "[WARN] No normals found in aggregate CSV; falling back to full velocity as shear proxy."
        )

    print("[INFO] Loading timestep data and matching to aggregate points...")
    agg_coords = agg[["x", "y", "z"]].values
    p, tau = load_timesteps_matched(
        timestep_dir, agg_coords, mu=mu, delta_n=delta_n, normals=normals
    )

    if p.shape[1] != len(agg):
        raise RuntimeError(
            "Mismatch between aggregate points and timestep data: %d vs %d" % (len(agg), p.shape[1])
        )

    print("[INFO] Computing corrected von Mises stress...")
    von_mises_corrected = compute_von_mises_equivalent(p, tau)

    agg["von_mises"] = von_mises_corrected
    agg.attrs = {
        "von_mises_definition": "Equivalent von Mises wall stress (pressure + shear, RMS, thin-wall)",
        "mu": mu,
        "delta_n": delta_n,
        "used_normals": normals is not None,
    }

    agg.to_csv(agg_file, index=False)

    print("[DONE] von Mises stress corrected successfully.")
    print(f"[OK] Updated file: {agg_file}")


if __name__ == "__main__":
    main()
