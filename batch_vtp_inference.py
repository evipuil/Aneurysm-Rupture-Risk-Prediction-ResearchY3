# Version 1 source snapshot
"""
batch_vtp_inference.py

Batch pipeline for processing VTP files and running DeepONet inference.
This script handles the complete pipeline from raw VTP geometry files to
predicted flow fields (x, y, z, pressure, u, v, w) saved as CSV files.

Pipeline:
1. Load VTP file (surface mesh)
2. Generate interior points within the geometry
3. Compute wall distances (SDF) for each interior point
4. Identify inlet boundary and compute inlet features
5. Run DeepONet inference
6. Save results as CSV (and optionally NPY/VTP)

Usage:
    python batch_vtp_inference.py --vtp-dir vtp_data --output-dir predictions
    python batch_vtp_inference.py --vtp-file vtp_data/C0001_cut1.vtp --output-dir predictions
    python batch_vtp_inference.py --vtp-dir vtp_data --output-dir predictions --flow-rate 0.2 --n-points 10000
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.cuda.amp import autocast
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista is required. Install with: pip install pyvista")
    sys.exit(1)

# Import neural network definitions
try:
    import neural_networks as nn_net
except ImportError:
    # Try from old_scripts
    sys.path.insert(0, str(Path(__file__).parent / "old_scripts"))
    import neural_networks as nn_net

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Default paths
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / "cfd_opt_deeponet" / "checkpoint" / "deeponet"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "predictions" / "batch_inference"


def setup_logger(log_dir: Path, name: str = "BatchVTPInference") -> logging.Logger:
    """
    Configure and return a logger for batch inference.

    Args:
        log_dir: Directory to save log file
        name: Logger name

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_dir / "batch_inference.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def compute_normals(points: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal vectors for each point by analyzing local neighborhood.

    Args:
        points: Array of point coordinates (N, 3)
        k: Number of neighbors for neighborhood-based normal calculation

    Returns:
        Tuple of (center_point, center_normal)
    """
    if len(points) < 3:
        centre = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
        return centre, np.array([0.0, 0.0, 1.0])

    k = min(k, max(len(points) - 1, 1))
    tree = KDTree(points)
    _, indices = tree.query(points, k=k + 1)
    normals = []

    for i in range(len(points)):
        neighborhood = points[indices[i]]
        centered_points = neighborhood - np.mean(neighborhood, axis=0)
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Eigenvector with smallest eigenvalue is the normal direction
        normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal) + 1e-12
        normals.append(normal)

    centre_points = np.mean(points, axis=0)
    centre_normals = np.mean(normals, axis=0)

    return centre_points, centre_normals


def extract_boundary_loops(mesh: pv.PolyData) -> List[pv.PolyData]:
    """
    Identify boundary loops in an open surface mesh.

    Args:
        mesh: PyVista PolyData surface mesh

    Returns:
        List of PolyData objects representing separate boundary loops
    """
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    if edges.n_points == 0:
        return []

    connectivity = edges.connectivity()
    labels = np.unique(connectivity["RegionId"])
    loops = [connectivity.threshold([label - 0.1, label + 0.1]) for label in labels]

    return loops


def build_inlet_cap(loop: pv.PolyData) -> pv.PolyData:
    """
    Build a triangulated cap from a boundary loop.

    Args:
        loop: PolyData representing a boundary loop

    Returns:
        Triangulated cap PolyData
    """
    if loop.n_points < 3:
        return loop
    try:
        cap = pv.PolyData(loop.points).delaunay_2d()
        return cap
    except Exception:
        return loop


def generate_interior_points(surface_mesh: pv.PolyData, n_points: int = 10000) -> np.ndarray:
    """
    Generate interior points within a closed or partially closed surface mesh.

    Args:
        surface_mesh: Surface mesh (PolyData)
        n_points: Target number of interior points

    Returns:
        Array of interior point coordinates (N, 3)
    """
    print(f"Generating {n_points} interior points from surface mesh...")

    # Try to fill holes in the mesh to make it watertight
    try:
        filled_mesh = surface_mesh.fill_holes(hole_size=1000.0)
    except Exception:
        filled_mesh = surface_mesh

    # Get bounding box
    bounds = filled_mesh.bounds

    # Generate random points within bounding box with extra candidates
    n_candidates = n_points * 20
    np.random.seed(SEED)
    x = np.random.uniform(bounds[0], bounds[1], n_candidates)
    y = np.random.uniform(bounds[2], bounds[3], n_candidates)
    z = np.random.uniform(bounds[4], bounds[5], n_candidates)
    candidate_points = np.column_stack([x, y, z])

    # Filter points that are inside the mesh
    candidate_cloud = pv.PolyData(candidate_points)

    interior_points = None
    try:
        selected = candidate_cloud.select_enclosed_points(
            filled_mesh, tolerance=0.0, check_surface=False
        )
        mask = selected["SelectedPoints"].astype(bool)
        interior_points = candidate_points[mask]
    except Exception:
        pass

    # Fallback: use distance-based filtering if enclosed selection fails
    if interior_points is None or len(interior_points) < n_points // 10:
        print("Using distance-based filtering as fallback...")
        tree = KDTree(filled_mesh.points)
        distances, _ = tree.query(candidate_points, k=1)
        # Points with moderate distance to surface are likely inside
        threshold = np.percentile(distances, 50)
        interior_points = candidate_points[distances < threshold]

    if len(interior_points) < n_points:
        print(f"Warning: Only {len(interior_points)} interior points found (requested {n_points})")
        # If we don't have enough points, add some surface points
        if len(interior_points) < n_points // 2:
            n_surface = min(n_points - len(interior_points), len(filled_mesh.points))
            surface_idx = np.random.choice(len(filled_mesh.points), n_surface, replace=False)
            additional_points = filled_mesh.points[surface_idx]
            if len(interior_points) > 0:
                interior_points = np.vstack([interior_points, additional_points])
            else:
                interior_points = additional_points
    else:
        # Randomly subsample to requested number
        idx = np.random.choice(len(interior_points), n_points, replace=False)
        interior_points = interior_points[idx]

    print(f"Final count: {len(interior_points)} interior points")
    return interior_points.astype(np.float32)


def preprocess_vtp(
    vtp_file: Path, n_points: int = 10000, flow_rate: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess a VTP file into DeepONet-compatible format.

    Args:
        vtp_file: Path to VTP file
        n_points: Number of interior points to generate
        flow_rate: Flow rate parameter (default 0.2 corresponds to m=0.002)

    Returns:
        Tuple of (X_sup, Y_sup, X_inlet, Simple_inlet)
        - X_sup: (1, N, 4) array of [x, y, z, sdf]
        - Y_sup: (1, N, 4) array of zeros (placeholder)
        - X_inlet: (1, M, 3) array of inlet point coordinates
        - Simple_inlet: (1, 7) array of [cx, cy, cz, nx, ny, nz, flow_rate]
    """
    # Load mesh
    mesh = pv.read(vtp_file)

    # Handle different mesh types
    if isinstance(mesh, pv.UnstructuredGrid):
        surface_mesh = mesh.extract_surface()
    elif isinstance(mesh, pv.PolyData):
        surface_mesh = mesh
    else:
        surface_mesh = mesh.extract_surface()

    # Generate interior points
    X_internal = generate_interior_points(surface_mesh, n_points)

    # Find boundary loops for inlet identification
    loops = extract_boundary_loops(surface_mesh)

    if len(loops) == 0:
        print("Warning: No boundary loops found, using all boundary points as inlet")
        # Fallback: use all boundary edges as inlet
        edges = surface_mesh.extract_feature_edges(boundary_edges=True)
        inlet_points = edges.points if edges.n_points > 0 else surface_mesh.points[:100]
        inlet_cap = pv.PolyData(inlet_points)
    else:
        # Build caps for each loop and select the largest as inlet
        caps = [build_inlet_cap(loop) for loop in loops]
        inlet_cap = max(
            caps, key=lambda c: c.area if hasattr(c, "area") and c.area > 0 else c.n_points
        )

    # Get wall points (entire surface)
    wall_points = surface_mesh.points

    # Calculate wall distance (SDF) for each interior point
    tree = KDTree(wall_points)
    distances, _ = tree.query(X_internal, k=1)
    sdf = distances.reshape(-1, 1)

    # Build X_sup: [x, y, z, sdf] with batch dimension
    X_sup = np.concatenate([X_internal, sdf], axis=-1).astype(np.float32)
    X_sup = X_sup[np.newaxis, ...]  # Shape: (1, N, 4)

    # Y_sup: placeholder zeros (unknown for inference)
    Y_sup = np.zeros((1, X_internal.shape[0], 4), dtype=np.float32)

    # X_inlet: inlet points with batch dimension
    X_inlet = inlet_cap.points.copy().astype(np.float32)
    X_inlet = X_inlet[np.newaxis, ...]  # Shape: (1, M, 3)

    # Compute inlet center and normal
    centre_inlet, normal_inlet = compute_normals(inlet_cap.points, k=10)

    # Orient normal toward the interior
    volume_center = X_internal.mean(axis=0)
    if np.dot(normal_inlet, (volume_center - centre_inlet)) < 0:
        normal_inlet = -normal_inlet

    # Build Simple_inlet: [cx, cy, cz, nx, ny, nz, flow_rate]
    simple_inlet = np.concatenate([centre_inlet, normal_inlet, [flow_rate]], axis=-1)
    simple_inlet = simple_inlet.astype(np.float32).reshape(1, -1)  # Shape: (1, 7)

    return X_sup, Y_sup, X_inlet, simple_inlet


def load_deeponet_models(
    checkpoint_dir: Path, device: torch.device, checkpoint_id: int = 5000
) -> Tuple[nn_net.Trunk, nn_net.Branch, nn_net.Branch_Bypass]:
    """
    Load DeepONet model components from checkpoint.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        device: Torch device
        checkpoint_id: Checkpoint iteration number

    Returns:
        Tuple of (trunk_net, branch_bc_net, branch_bp_net)
    """
    # Model architecture parameters
    bc_dim = 64
    hidden_num = bc_dim
    out_dim = int(4 * hidden_num)
    layer_num = 4
    in_dim = 4

    # Initialize networks
    trunk_net = nn_net.Trunk(in_dim, out_dim, hidden_num, layer_num).to(device)
    branch_bc_net = nn_net.Branch(7, bc_dim, bc_dim, 4).to(device)
    branch_bp_net = nn_net.Branch_Bypass(1, 4).to(device)

    # Load weights
    trunk_state = torch.load(checkpoint_dir / f"trunk_{checkpoint_id}", map_location=device)
    trunk_state = {k.replace("_orig_mod.", ""): v for k, v in trunk_state.items()}
    trunk_net.load_state_dict(trunk_state)

    branch_bc_state = torch.load(checkpoint_dir / f"branch_bc_{checkpoint_id}", map_location=device)
    branch_bc_state = {k.replace("_orig_mod.", ""): v for k, v in branch_bc_state.items()}
    branch_bc_net.load_state_dict(branch_bc_state)

    branch_bp_state = torch.load(checkpoint_dir / f"branch_bp_{checkpoint_id}", map_location=device)
    branch_bp_state = {k.replace("_orig_mod.", ""): v for k, v in branch_bp_state.items()}
    branch_bp_net.load_state_dict(branch_bp_state)

    # Set to eval mode
    trunk_net.eval()
    branch_bc_net.eval()
    branch_bp_net.eval()

    return trunk_net, branch_bc_net, branch_bp_net


def run_deeponet_inference(
    X_sup: np.ndarray,
    Simple_inlet: np.ndarray,
    trunk_net: nn_net.Trunk,
    branch_bc_net: nn_net.Branch,
    branch_bp_net: nn_net.Branch_Bypass,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> np.ndarray:
    """
    Run DeepONet inference to predict flow field.

    Args:
        X_sup: Input coordinates with SDF (1, N, 4)
        Simple_inlet: Inlet features (1, 7)
        trunk_net: Trunk network
        branch_bc_net: Branch BC network
        branch_bp_net: Branch bypass network
        device: Torch device
        use_mixed_precision: Whether to use FP16

    Returns:
        Predictions array (N, 4) containing [p, u, v, w]
    """
    # Convert to tensors
    X = torch.tensor(X_sup, dtype=torch.float32, device=device)
    X_in = torch.tensor(Simple_inlet, dtype=torch.float32, device=device)

    with torch.no_grad():
        if use_mixed_precision and device.type == "cuda":
            with autocast():
                trunk_pred1, trunk_pred2, trunk_pred3, trunk_pred4 = trunk_net(X)
                branch_bc_pred = branch_bc_net(X_in).unsqueeze(-1)

                h_pred1 = torch.matmul(trunk_pred1, branch_bc_pred)
                h_pred2 = torch.matmul(trunk_pred2, branch_bc_pred)
                h_pred3 = torch.matmul(trunk_pred3, branch_bc_pred)
                h_pred4 = torch.matmul(trunk_pred4, branch_bc_pred)

                y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)
                branch_bp_pred = branch_bp_net(X_in[..., -1:])
                y_pred = y_pred * branch_bp_pred.unsqueeze(1)
        else:
            trunk_pred1, trunk_pred2, trunk_pred3, trunk_pred4 = trunk_net(X)
            branch_bc_pred = branch_bc_net(X_in).unsqueeze(-1)

            h_pred1 = torch.matmul(trunk_pred1, branch_bc_pred)
            h_pred2 = torch.matmul(trunk_pred2, branch_bc_pred)
            h_pred3 = torch.matmul(trunk_pred3, branch_bc_pred)
            h_pred4 = torch.matmul(trunk_pred4, branch_bc_pred)

            y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)
            branch_bp_pred = branch_bp_net(X_in[..., -1:])
            y_pred = y_pred * branch_bp_pred.unsqueeze(1)

    # Convert to numpy and squeeze batch dimension
    predictions = y_pred[0].cpu().numpy()  # Shape: (N, 4) -> [p, u, v, w]

    # Post-process pressure (zero-mean)
    predictions[:, 0] = predictions[:, 0] - np.mean(predictions[:, 0])

    # Post-process velocity (add 0.5 to u component as done in training)
    predictions[:, 1] = predictions[:, 1] + 0.5

    return predictions


def save_results_csv(points: np.ndarray, predictions: np.ndarray, output_file: Path):
    """
    Save results as CSV file with columns: x, y, z, p, u, v, w

    Args:
        points: Coordinates (N, 3)
        predictions: Flow predictions (N, 4) - [p, u, v, w]
        output_file: Output CSV path
    """
    # Combine coordinates and predictions
    data = np.concatenate([points, predictions], axis=1)

    # Save with header
    np.savetxt(output_file, data, delimiter=",", header="x,y,z,p,u,v,w", comments="")


def save_results_npy(points: np.ndarray, predictions: np.ndarray, output_file: Path):
    """
    Save results as NPY file with shape (N, 7) - [x, y, z, p, u, v, w]

    Args:
        points: Coordinates (N, 3)
        predictions: Flow predictions (N, 4) - [p, u, v, w]
        output_file: Output NPY path
    """
    data = np.concatenate([points, predictions], axis=1)
    np.save(output_file, data)


def save_results_vtp(points: np.ndarray, predictions: np.ndarray, output_file: Path):
    """
    Save results as VTP file with point data arrays for visualization.

    Args:
        points: Coordinates (N, 3)
        predictions: Flow predictions (N, 4) - [p, u, v, w]
        output_file: Output VTP path
    """
    cloud = pv.PolyData(points)
    cloud["Pressure"] = predictions[:, 0]
    cloud["Velocity"] = predictions[:, 1:4]
    cloud["VelocityMagnitude"] = np.linalg.norm(predictions[:, 1:4], axis=1)
    cloud.save(output_file)


def process_single_vtp(
    vtp_file: Path,
    output_dir: Path,
    trunk_net: nn_net.Trunk,
    branch_bc_net: nn_net.Branch,
    branch_bp_net: nn_net.Branch_Bypass,
    device: torch.device,
    n_points: int = 10000,
    flow_rate: float = 0.2,
    save_npy: bool = False,
    save_vtp: bool = False,
    use_mixed_precision: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Process a single VTP file through the complete pipeline.

    Args:
        vtp_file: Input VTP file path
        output_dir: Output directory
        trunk_net, branch_bc_net, branch_bp_net: Model networks
        device: Torch device
        n_points: Number of interior points
        flow_rate: Flow rate parameter
        save_npy: Whether to also save NPY file
        save_vtp: Whether to also save VTP file
        use_mixed_precision: Whether to use FP16
        logger: Logger instance

    Returns:
        Dictionary with processing results and timing
    """
    log = logger.info if logger else print

    case_name = vtp_file.stem
    start_time = time.time()

    log(f"Processing: {vtp_file.name}")

    # Step 1: Preprocess VTP
    preprocess_start = time.time()
    X_sup, Y_sup, X_inlet, Simple_inlet = preprocess_vtp(vtp_file, n_points, flow_rate)
    preprocess_time = time.time() - preprocess_start
    log(f"  Preprocessing: {preprocess_time:.2f}s, {X_sup.shape[1]} points")

    # Step 2: Run inference
    inference_start = time.time()
    predictions = run_deeponet_inference(
        X_sup, Simple_inlet, trunk_net, branch_bc_net, branch_bp_net, device, use_mixed_precision
    )
    inference_time = time.time() - inference_start
    log(f"  Inference: {inference_time:.4f}s")

    # Extract coordinates (without SDF)
    points = X_sup[0, :, :3]

    # Step 3: Save results
    save_start = time.time()

    # Always save CSV
    csv_file = output_dir / f"{case_name}.csv"
    save_results_csv(points, predictions, csv_file)

    # Optionally save NPY
    if save_npy:
        npy_file = output_dir / f"{case_name}.npy"
        save_results_npy(points, predictions, npy_file)

    # Optionally save VTP
    if save_vtp:
        vtp_output_file = output_dir / f"{case_name}_predicted.vtp"
        save_results_vtp(points, predictions, vtp_output_file)

    save_time = time.time() - save_start
    total_time = time.time() - start_time

    log(f"  Saved: {csv_file.name} (total: {total_time:.2f}s)")

    return {
        "case_name": case_name,
        "n_points": X_sup.shape[1],
        "preprocess_time": preprocess_time,
        "inference_time": inference_time,
        "save_time": save_time,
        "total_time": total_time,
        "output_csv": str(csv_file),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch VTP to DeepONet inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all VTP files in a directory:
    python batch_vtp_inference.py --vtp-dir vtp_data --output-dir predictions

  Process a single VTP file:
    python batch_vtp_inference.py --vtp-file vtp_data/C0001_cut1.vtp --output-dir predictions

  Process with custom settings:
    python batch_vtp_inference.py --vtp-dir vtp_data --output-dir predictions --n-points 20000 --flow-rate 0.3 --save-vtp
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--vtp-dir", type=str, help="Directory containing VTP files")
    input_group.add_argument("--vtp-file", type=str, help="Single VTP file to process")

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )

    # Model options
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help=f"DeepONet checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=int,
        default=5000,
        help="Checkpoint iteration to load (default: 5000)",
    )

    # Processing options
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000,
        help="Number of interior points to generate (default: 10000)",
    )
    parser.add_argument(
        "--flow-rate",
        type=float,
        default=0.2,
        help="Flow rate parameter (default: 0.2, corresponds to m=0.002)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Use mixed precision (FP16) for inference"
    )

    # Output format options
    parser.add_argument("--save-npy", action="store_true", help="Also save results as NPY files")
    parser.add_argument(
        "--save-vtp", action="store_true", help="Also save results as VTP files for visualization"
    )

    # Other options
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of files to process (for testing)"
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Setup logger
    logger = setup_logger(output_dir)
    logger.info("=" * 60)
    logger.info("Batch VTP Inference Pipeline")
    logger.info("=" * 60)

    # Get list of VTP files
    if args.vtp_file:
        vtp_files = [Path(args.vtp_file)]
        if not vtp_files[0].exists():
            logger.error(f"VTP file not found: {args.vtp_file}")
            sys.exit(1)
    else:
        vtp_dir = Path(args.vtp_dir)
        if not vtp_dir.exists():
            logger.error(f"VTP directory not found: {args.vtp_dir}")
            sys.exit(1)
        vtp_files = sorted(vtp_dir.glob("*.vtp"))
        if not vtp_files:
            logger.error(f"No VTP files found in: {args.vtp_dir}")
            sys.exit(1)

    # Apply limit if specified
    if args.limit:
        vtp_files = vtp_files[: args.limit]

    logger.info(f"Found {len(vtp_files)} VTP files to process")
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Interior points: {args.n_points}")
    logger.info(f"Flow rate: {args.flow_rate}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info("=" * 60)

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load models
    logger.info("Loading DeepONet models...")
    try:
        trunk_net, branch_bc_net, branch_bp_net = load_deeponet_models(
            checkpoint_dir, device, args.checkpoint_id
        )
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        sys.exit(1)

    # Process files
    results = []
    failed = []

    for vtp_file in tqdm(vtp_files, desc="Processing VTP files"):
        try:
            result = process_single_vtp(
                vtp_file=vtp_file,
                output_dir=output_dir,
                trunk_net=trunk_net,
                branch_bc_net=branch_bc_net,
                branch_bp_net=branch_bp_net,
                device=device,
                n_points=args.n_points,
                flow_rate=args.flow_rate,
                save_npy=args.save_npy,
                save_vtp=args.save_vtp,
                use_mixed_precision=args.mixed_precision,
                logger=logger,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {vtp_file.name}: {e}")
            failed.append({"file": str(vtp_file), "error": str(e)})

    # Save summary
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Successful: {len(results)}/{len(vtp_files)}")
    logger.info(f"Failed: {len(failed)}")

    if results:
        avg_time = np.mean([r["total_time"] for r in results])
        avg_inference = np.mean([r["inference_time"] for r in results])
        logger.info(f"Average total time: {avg_time:.2f}s")
        logger.info(f"Average inference time: {avg_inference:.4f}s")

    # Save summary CSV
    summary_file = output_dir / "processing_summary.csv"
    with open(summary_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    logger.info(f"Summary saved to: {summary_file}")

    # Save failed list if any
    if failed:
        failed_file = output_dir / "failed_files.csv"
        with open(failed_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "error"])
            writer.writeheader()
            writer.writerows(failed)
        logger.info(f"Failed files logged to: {failed_file}")

    logger.info("Pipeline completed!")


if __name__ == "__main__":
    main()
