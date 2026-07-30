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
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "old_scripts"))

# Configuration
VTP_DIR = "vtp_data"
OUTPUT_DIR = "predictions/batch_results"
CHECKPOINT_DIR = "cfd_opt_deeponet/checkpoint/deeponet"
CHECKPOINT_ID = 5000
N_POINTS = 10000
FLOW_RATE = 0.2
SAVE_NPY = False
SAVE_VTP = False
USE_MIXED_PRECISION = False
LIMIT = None
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
log_dir = Path(OUTPUT_DIR)
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("VTPInference")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_file = logging.FileHandler(log_dir / "batch_inference.log")
_file.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_console)
logger.addHandler(_file)

# Geometry helpers


def compute_normals(points, k=10):
    # Estimate a representative center and average normal from local PCA
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
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        n = eigenvectors[:, 0]
        normals.append(n / (np.linalg.norm(n) + 1e-12))
    return np.mean(points, axis=0), np.mean(normals, axis=0)


def extract_boundary_loops(mesh):
    # Find open boundary edges and split into connected components
    edges = mesh.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    )
    if edges.n_points == 0:
        return []
    conn = edges.connectivity()
    labels = np.unique(conn["RegionId"])
    return [conn.threshold([lb - 0.1, lb + 0.1]) for lb in labels]


def build_inlet_cap(loop):
    # Triangulate a boundary loop to form a cap
    if loop.n_points < 3:
        return loop
    try:
        return pv.PolyData(loop.points).delaunay_2d()
    except Exception:
        return loop


def generate_interior_points(surface_mesh, n_points):
    # Fill holes, scatter random candidates inside the bounding box, keep interior ones
    logger.info(f"Generating {n_points} interior points ...")
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

    # Fallback: distance-based heuristic
    if interior is None or len(interior) < n_points // 10:
        logger.info("Using distance-based fallback for interior points")
        tree = KDTree(filled.points)
        dists, _ = tree.query(candidates, k=1)
        interior = candidates[dists < np.percentile(dists, 50)]

    if len(interior) < n_points:
        logger.warning(f"Only {len(interior)} interior pts found (wanted {n_points})")
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

    logger.info(f"Final interior points: {len(interior)}")
    return interior.astype(np.float32)


# VTP preprocessing (geometry -> DeepONet inputs)
def preprocess_vtp(vtp_file, n_points, flow_rate):
    mesh = pv.read(vtp_file)
    surface = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh

    X_internal = generate_interior_points(surface, n_points)

    # Identify inlet as the largest boundary loop cap
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

    # SDF (wall distance)
    tree = KDTree(surface.points)
    sdf = tree.query(X_internal, k=1)[0].reshape(-1, 1)

    X_sup = np.concatenate([X_internal, sdf], axis=-1).astype(np.float32)[np.newaxis]  # (1, N, 4)
    Y_sup = np.zeros((1, len(X_internal), 4), dtype=np.float32)
    X_inlet = inlet_cap.points.copy().astype(np.float32)[np.newaxis]  # (1, M, 3)

    # Inlet center, oriented normal, flow rate
    centre, normal = compute_normals(inlet_cap.points, k=10)
    vol_center = X_internal.mean(axis=0)
    if np.dot(normal, vol_center - centre) < 0:
        normal = -normal
    simple_inlet = np.concatenate([centre, normal, [flow_rate]]).astype(np.float32).reshape(1, -1)

    return X_sup, Y_sup, X_inlet, simple_inlet


# Load DeepONet checkpoint
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


# Run inference on prepared inputs
def run_inference(X_sup, simple_inlet, trunk, branch_bc, branch_bp, device):
    X = torch.tensor(X_sup, dtype=torch.float32, device=device)
    X_in = torch.tensor(simple_inlet, dtype=torch.float32, device=device)

    with torch.no_grad():
        t1, t2, t3, t4 = trunk(X)
        bc = branch_bc(X_in).unsqueeze(-1)
        h1 = torch.matmul(t1, bc)
        h2 = torch.matmul(t2, bc)
        h3 = torch.matmul(t3, bc)
        h4 = torch.matmul(t4, bc)
        y = torch.cat([h1, h2, h3, h4], dim=-1)
        bp = branch_bp(X_in[..., -1:])
        y = y * bp.unsqueeze(1)

    preds = y[0].cpu().numpy()  # (N, 4) -> [p, u, v, w]
    return preds


# Save functions


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


# Process one VTP file end-to-end
def process_one(vtp_file, trunk, branch_bc, branch_bp, device):
    case = Path(vtp_file).stem
    t0 = time.time()
    logger.info(f"Processing: {Path(vtp_file).name}")

    # Preprocess
    tp = time.time()
    X_sup, _, _, simple_inlet = preprocess_vtp(vtp_file, N_POINTS, FLOW_RATE)
    t_pre = time.time() - tp
    logger.info(f"  Preprocess: {t_pre:.2f}s, {X_sup.shape[1]} points")

    # Inference
    ti = time.time()
    preds = run_inference(X_sup, simple_inlet, trunk, branch_bc, branch_bp, device)
    t_inf = time.time() - ti
    logger.info(f"  Inference: {t_inf:.4f}s")

    # Save
    points = X_sup[0, :, :3]
    out = Path(OUTPUT_DIR)
    csv_path = out / f"{case}.csv"
    save_csv(points, preds, csv_path)
    if SAVE_NPY:
        save_npy(points, preds, out / f"{case}.npy")
    if SAVE_VTP:
        save_vtp(points, preds, out / f"{case}_predicted.vtp")

    total = time.time() - t0
    logger.info(f"  Saved: {csv_path.name} (total: {total:.2f}s)")

    return {
        "case_name": case,
        "n_points": X_sup.shape[1],
        "preprocess_time": t_pre,
        "inference_time": t_inf,
        "total_time": total,
        "output_csv": str(csv_path),
    }


# Gather VTP files
vtp_dir = Path(VTP_DIR)
if not vtp_dir.exists():
    logger.error(f"VTP directory not found: {vtp_dir}")
    sys.exit(1)
vtp_files = sorted(vtp_dir.glob("*.vtp"))
if not vtp_files:
    logger.error(f"No VTP files in {vtp_dir}")
    sys.exit(1)
if LIMIT:
    vtp_files = vtp_files[:LIMIT]

logger.info(f"Files: {len(vtp_files)}  |  Device: {DEVICE}")
logger.info(f"Checkpoint: {CHECKPOINT_DIR}  |  Points: {N_POINTS}  |  Flow rate: {FLOW_RATE}")
logger.info("=" * 60)

# Load models
try:
    trunk, branch_bc, branch_bp = load_deeponet(CHECKPOINT_DIR, DEVICE, CHECKPOINT_ID)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    sys.exit(1)

# Process each file
results, failed = [], []
for vf in vtp_files:
    try:
        results.append(process_one(vf, trunk, branch_bc, branch_bp, DEVICE))
    except Exception as e:
        logger.error(f"Failed {vf.name}: {e}")
        failed.append({"file": str(vf), "error": str(e)})

# Summary
logger.info("=" * 60)
logger.info("Processing Summary")
logger.info(f"Successful: {len(results)}/{len(vtp_files)}   Failed: {len(failed)}")
if results:
    avg_t = np.mean([r["total_time"] for r in results])
    avg_i = np.mean([r["inference_time"] for r in results])
    logger.info(f"Avg total time: {avg_t:.2f}s   Avg inference: {avg_i:.4f}s")

# Save processing summary CSV
summary_path = Path(OUTPUT_DIR) / "processing_summary.csv"
if results:
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    logger.info(f"Summary saved to {summary_path}")

if failed:
    fail_path = Path(OUTPUT_DIR) / "failed_files.csv"
    with open(fail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "error"])
        w.writeheader()
        w.writerows(failed)
    logger.info(f"Failed files logged to {fail_path}")

logger.info("Pipeline complete!")
