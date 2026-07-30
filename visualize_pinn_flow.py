# Version 14 source snapshot
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

try:
    import pyvista as pv
except Exception:
    pv = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = (
    ROOT / "predictions" / "pinn_outlet_loss_sample_1000_full" / "ANSYS_UNIGE_09_cut1"
)
DEFAULT_OUTPUT_DIR = ROOT / "results_v14_pinn_flow_field"
MAIN_TITLE = {"fontsize": 14, "fontweight": "bold", "fontfamily": "Times New Roman"}
PANEL_TITLE = {"fontsize": 12, "fontweight": "bold", "fontfamily": "Times New Roman"}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def _set_equal_3d_axes(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1.0
    half = span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _natural_timestep_key(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def find_flow_files(case_dir: Path) -> list[Path]:
    timesteps = case_dir / "timesteps"
    files = sorted(timesteps.glob("flow_t*.csv"), key=_natural_timestep_key)
    if files:
        return files

    steady = timesteps / "flow_steady.csv"
    if steady.exists():
        return [steady]

    raise FileNotFoundError(f"No flow_t*.csv or flow_steady.csv files found under {timesteps}")


def read_flow(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"x", "y", "z", "u", "v", "w"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    for col in ["x", "y", "z", "p", "u", "v", "w"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["x", "y", "z", "u", "v", "w"]).copy()
    frame["speed"] = np.linalg.norm(frame[["u", "v", "w"]].to_numpy(dtype=float), axis=1)
    return frame


def read_scalar_csv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=sorted(required)).copy()


def read_vtp_points(path: Path) -> np.ndarray | None:
    if not path.exists() or pv is None:
        return None
    mesh = pv.read(path)
    if mesh.n_points == 0:
        return None
    return np.asarray(mesh.points, dtype=float)


def draw_vtp_context(
    ax, vtp_points: np.ndarray | None, alpha: float, point_size: float = 12.0
) -> None:
    if vtp_points is None or len(vtp_points) == 0 or alpha <= 0:
        return
    ax.scatter(
        vtp_points[:, 0],
        vtp_points[:, 1],
        vtp_points[:, 2],
        c="#6B6B6B",
        s=point_size,
        alpha=alpha,
        linewidths=0,
        depthshade=False,
    )


def read_time_labels(case_dir: Path) -> dict[str, str]:
    path = case_dir / "timesteps" / "time_index.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "timestep" not in frame.columns or "time_seconds" not in frame.columns:
        return {}
    labels: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        try:
            idx = int(float(getattr(row, "timestep")))
            time_s = float(getattr(row, "time_seconds"))
        except Exception:
            continue
        labels[f"flow_t{idx:02d}"] = f"t = {time_s:.2f} s"
    return labels


def choose_timestep(files: list[Path], value: str) -> Path:
    if value.lower() in {"mid", "middle", "representative"}:
        return files[len(files) // 2]
    if value.lower() in {"peak", "max"}:
        means = []
        for path in files:
            frame = read_flow(path)
            means.append(float(frame["speed"].mean()))
        return files[int(np.argmax(means))]
    idx = int(value)
    if idx < 0 or idx >= len(files):
        raise IndexError(f"Timestep index {idx} is outside available range 0-{len(files) - 1}")
    return files[idx]


def deterministic_subset(n_rows: int, n_keep: int) -> np.ndarray:
    if n_rows <= n_keep:
        return np.arange(n_rows)
    return np.linspace(0, n_rows - 1, n_keep, dtype=int)


def draw_flow_panel(
    ax,
    frame: pd.DataFrame,
    title: str,
    norm: Normalize,
    point_limit: int,
    arrow_count: int,
    vtp_points: np.ndarray | None = None,
    vtp_alpha: float = 0.0,
    cmap: str = "viridis",
) -> None:
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=float)
    uvw = frame[["u", "v", "w"]].to_numpy(dtype=float)
    speed = frame["speed"].to_numpy(dtype=float)

    draw_vtp_context(ax, vtp_points, vtp_alpha)

    point_idx = deterministic_subset(len(frame), point_limit)
    ax.scatter(
        xyz[point_idx, 0],
        xyz[point_idx, 1],
        xyz[point_idx, 2],
        c=speed[point_idx],
        cmap=cmap,
        norm=norm,
        s=8,
        alpha=0.72,
        linewidths=0,
    )

    if len(frame) > 0 and arrow_count > 0:
        arrow_idx = np.argsort(speed)[-min(arrow_count, len(frame)) :]
        arrow_idx = arrow_idx[
            np.linspace(0, len(arrow_idx) - 1, min(arrow_count, len(arrow_idx)), dtype=int)
        ]
        span = max(
            float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), float(np.ptp(xyz[:, 2])), 1.0
        )
        arrow_length = span * 0.055
        colors = plt.get_cmap(cmap)(norm(speed[arrow_idx]))
        ax.quiver(
            xyz[arrow_idx, 0],
            xyz[arrow_idx, 1],
            xyz[arrow_idx, 2],
            uvw[arrow_idx, 0],
            uvw[arrow_idx, 1],
            uvw[arrow_idx, 2],
            length=arrow_length,
            normalize=True,
            colors=colors,
            linewidths=0.65,
            alpha=0.95,
        )

    _set_equal_3d_axes(ax, xyz)
    ax.view_init(elev=23, azim=-58)
    ax.set_title(title, pad=10, **PANEL_TITLE)
    ax.set_xlabel("x", labelpad=6)
    ax.set_ylabel("y", labelpad=6)
    ax.set_zlabel("z", labelpad=6)
    ax.grid(False)


def draw_scalar_panel(
    ax,
    frame: pd.DataFrame,
    value_col: str,
    title: str,
    norm: Normalize,
    point_limit: int,
    vtp_points: np.ndarray | None,
    vtp_alpha: float,
    zoom: float,
    cmap: str = "viridis",
) -> None:
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=float)
    values = frame[value_col].to_numpy(dtype=float)
    draw_vtp_context(ax, vtp_points, vtp_alpha)
    point_idx = deterministic_subset(len(frame), point_limit)
    ax.scatter(
        xyz[point_idx, 0],
        xyz[point_idx, 1],
        xyz[point_idx, 2],
        c=values[point_idx],
        cmap=cmap,
        norm=norm,
        s=12,
        alpha=0.86,
        linewidths=0,
        depthshade=False,
    )
    _set_equal_3d_axes(ax, xyz if vtp_points is None else np.vstack([xyz, vtp_points]))
    ax.view_init(elev=23, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, pad=8, **PANEL_TITLE)
    ax.set_axis_off()


def make_single_timestep_figure(
    flow_path: Path,
    out_dir: Path,
    label: str,
    point_limit: int,
    arrow_count: int,
    dpi: int,
    vtp_points: np.ndarray | None,
    vtp_alpha: float,
) -> Path:
    frame = read_flow(flow_path)
    norm = Normalize(vmin=float(frame["speed"].min()), vmax=float(frame["speed"].max()))
    fig = plt.figure(figsize=(9.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    draw_flow_panel(
        ax,
        frame,
        "Representative PINN Flow Field"
        if not label
        else f"Representative PINN Flow Field, {label}",
        norm=norm,
        point_limit=point_limit,
        arrow_count=arrow_count,
        vtp_points=vtp_points,
        vtp_alpha=vtp_alpha,
    )
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.70, pad=0.08)
    cbar.set_label("Velocity magnitude", fontsize=12)
    fig.tight_layout()

    out_path = out_dir / f"representative_pinn_flow_field_{flow_path.stem}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_snapshot_figure(
    files: list[Path],
    out_dir: Path,
    time_labels: dict[str, str],
    point_limit: int,
    arrow_count: int,
    dpi: int,
    n_snapshots: int,
    vtp_points: np.ndarray | None,
    vtp_alpha: float,
) -> Path:
    selected_idx = np.linspace(0, len(files) - 1, min(n_snapshots, len(files)), dtype=int)
    selected_files = [files[i] for i in selected_idx]
    frames = [read_flow(path) for path in selected_files]
    all_speed = np.concatenate([frame["speed"].to_numpy(dtype=float) for frame in frames])
    norm = Normalize(vmin=float(all_speed.min()), vmax=float(all_speed.max()))

    n_cols = min(5, len(frames))
    n_rows = int(np.ceil(len(frames) / n_cols))
    fig = plt.figure(figsize=(3.7 * n_cols, 3.7 * n_rows + 0.45))
    for i, (path, frame) in enumerate(zip(selected_files, frames), start=1):
        ax = fig.add_subplot(n_rows, n_cols, i, projection="3d")
        draw_flow_panel(
            ax,
            frame,
            time_labels.get(path.stem, path.stem),
            norm=norm,
            point_limit=point_limit,
            arrow_count=max(20, arrow_count // 3),
            vtp_points=vtp_points,
            vtp_alpha=vtp_alpha,
        )
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.zaxis.label.set_size(8)

    fig.suptitle("Representative PINN Flow Field Across the Cardiac Cycle", **MAIN_TITLE)
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"), ax=fig.axes, shrink=0.62, pad=0.02
    )
    cbar.set_label("Velocity magnitude", fontsize=12)
    out_path = out_dir / "representative_pinn_flow_field_snapshots.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_hemodynamic_grid(
    case_dir: Path,
    flow_path: Path,
    out_dir: Path,
    label: str,
    point_limit: int,
    dpi: int,
    vtp_points: np.ndarray | None,
    vtp_alpha: float,
    cmap: str,
    zoom: float,
    legacy_wss_scale: float,
    angiogram_path: Path | None,
) -> Path:
    flow = read_flow(flow_path)
    wss_path = flow_path.parent / flow_path.name.replace("flow_", "wss_")
    if not wss_path.exists():
        raise FileNotFoundError(f"Could not find matching WSS file for {flow_path}: {wss_path}")
    wss = read_scalar_csv(wss_path, {"x", "y", "z", "wss_magnitude"})
    aggregate = read_scalar_csv(
        case_dir / "hemodynamics_aggregate.csv", {"x", "y", "z", "tawss", "osi"}
    )

    units_path = case_dir / "hemodynamic_units.json"
    has_unit_metadata = units_path.exists()
    units = {}
    if has_unit_metadata:
        with open(units_path, "r", encoding="utf-8") as handle:
            units = json.load(handle)
    wss_scale = 1.0 if has_unit_metadata else float(legacy_wss_scale)
    wss["wss_magnitude"] *= wss_scale
    aggregate["tawss"] *= wss_scale

    osi = np.clip(aggregate["osi"].to_numpy(dtype=float), 0.0, 0.499)
    tawss = np.maximum(aggregate["tawss"].to_numpy(dtype=float), 1e-6)
    denom = np.maximum((1.0 - 2.0 * osi) * tawss, 1e-6)
    aggregate["osi"] = osi
    aggregate["rrt"] = 1.0 / denom

    pressure_unit = str(units.get("pressure_unit", "")).strip()
    pressure_label = (
        f"Relative pressure ({pressure_unit})"
        if pressure_unit
        else "Relative pressure (normalized)"
    )
    panels = [
        (pressure_label, flow, "p"),
        ("Velocity magnitude (m/s)", flow, "speed"),
        ("Instantaneous WSS (Pa)", wss, "wss_magnitude"),
        ("Time-averaged WSS (Pa)", aggregate, "tawss"),
        ("OSI (dimensionless)", aggregate, "osi"),
        ("RRT (Pa$^{-1}$)", aggregate, "rrt"),
    ]

    # Use one physical color scale for instantaneous and time-averaged WSS.
    # Independent normalization can make two related fields look identical even
    # when their absolute values differ.
    shared_wss = np.concatenate(
        [
            wss["wss_magnitude"].to_numpy(dtype=float),
            aggregate["tawss"].to_numpy(dtype=float),
        ]
    )
    shared_wss = shared_wss[np.isfinite(shared_wss)]
    shared_wss_limits = (
        tuple(np.nanpercentile(shared_wss, [1.0, 99.0])) if len(shared_wss) else (0.0, 1.0)
    )

    has_angiogram = angiogram_path is not None and angiogram_path.exists()
    if has_angiogram:
        fig = plt.figure(figsize=(17.2, 9.6))
        grid = fig.add_gridspec(2, 4, width_ratios=[0.78, 1.0, 1.0, 1.0], wspace=0.16, hspace=0.30)
        angiogram_ax = fig.add_subplot(grid[:, 0])
        with Image.open(angiogram_path) as angiogram:
            angiogram_ax.imshow(angiogram.convert("L"), cmap="gray")
        angiogram_ax.set_title("Illustrative cerebral angiogram", pad=8, **PANEL_TITLE)
        angiogram_ax.text(
            0.5,
            -0.025,
            "Reference image; not the modeled case\nLucien Monfils, CC BY-SA 3.0",
            transform=angiogram_ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        angiogram_ax.axis("off")
        panel_positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    else:
        fig = plt.figure(figsize=(14.4, 10.8))
        grid = fig.add_gridspec(2, 3, wspace=0.16, hspace=0.34)
        panel_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for i, (title, frame, col) in enumerate(panels, start=1):
        values = frame[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if col in {"wss_magnitude", "tawss"}:
            vmin, vmax = shared_wss_limits
        else:
            vmin, vmax = np.nanpercentile(finite, [1.0, 99.0]) if len(finite) else (0.0, 1.0)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
        row, col_idx = panel_positions[i - 1]
        ax = fig.add_subplot(grid[row, col_idx], projection="3d")
        draw_scalar_panel(
            ax,
            frame,
            col,
            title,
            norm=norm,
            point_limit=point_limit,
            vtp_points=vtp_points,
            vtp_alpha=vtp_alpha,
            zoom=zoom,
            cmap=cmap,
        )
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.zaxis.label.set_size(9)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.62, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    suptitle = (
        "Representative PINN Hemodynamic Fields: Transient (Top) and Time-Aggregated (Bottom)"
    )
    if label:
        suptitle = f"{suptitle}, {label}"
    fig.suptitle(suptitle, **MAIN_TITLE)
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.075, top=0.90)

    out_path = out_dir / f"representative_pinn_six_hemodynamic_fields_{flow_path.stem}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def maybe_make_gif(
    files: list[Path], out_dir: Path, point_limit: int, arrow_count: int, dpi: int
) -> Path:
    frames = [read_flow(path) for path in files]
    all_speed = np.concatenate([frame["speed"].to_numpy(dtype=float) for frame in frames])
    norm = Normalize(vmin=float(all_speed.min()), vmax=float(all_speed.max()))
    temp_paths: list[Path] = []

    for path, frame in zip(files, frames):
        fig = plt.figure(figsize=(7.2, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        draw_flow_panel(
            ax,
            frame,
            path.stem,
            norm=norm,
            point_limit=point_limit,
            arrow_count=arrow_count,
        )
        fig.tight_layout()
        temp_path = out_dir / f"_gif_{path.stem}.png"
        fig.savefig(temp_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        temp_paths.append(temp_path)

    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in temp_paths]
    gif_path = out_dir / "representative_pinn_flow_field_animation.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=350, loop=0)
    for image in images:
        image.close()
    for path in temp_paths:
        path.unlink(missing_ok=True)
    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the representative PINN flow field from saved timestep CSVs."
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timestep", default="peak", help="Timestep index, 'mid', or 'peak'. Default: peak"
    )
    parser.add_argument(
        "--point-limit", type=int, default=12000, help="Maximum points shown as colored scatter."
    )
    parser.add_argument(
        "--arrow-count", type=int, default=220, help="Number of direction arrows overlaid."
    )
    parser.add_argument(
        "--snapshots", type=int, default=10, help="Number of cardiac-cycle snapshots to include."
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--vtp-path",
        type=Path,
        default=None,
        help="Optional VTP context file. Defaults to case hemodynamics_aggregate.vtp",
    )
    parser.add_argument(
        "--vtp-alpha", type=float, default=0.2, help="Alpha for the translucent VTP context points."
    )
    parser.add_argument(
        "--hemo-cmap", default="viridis", help="Shared colormap for all six hemodynamic panels."
    )
    parser.add_argument(
        "--hemo-zoom",
        type=float,
        default=1.55,
        help="3D camera zoom for the six hemodynamic panels.",
    )
    parser.add_argument(
        "--legacy-wss-scale",
        type=float,
        default=1000.0,
        help=(
            "Scale legacy WSS/TAWSS outputs from per-mm gradients to Pa. "
            "Ignored when hemodynamic_units.json exists."
        ),
    )
    parser.add_argument(
        "--angiogram-path",
        type=Path,
        default=None,
        help="Optional illustrative angiogram shown beside the six fields.",
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    flow_files = find_flow_files(case_dir)
    time_labels = read_time_labels(case_dir)
    selected = choose_timestep(flow_files, args.timestep)
    vtp_path = (args.vtp_path or (case_dir / "hemodynamics_aggregate.vtp")).resolve()
    vtp_points = read_vtp_points(vtp_path)

    single = make_single_timestep_figure(
        selected,
        out_dir,
        time_labels.get(selected.stem, ""),
        args.point_limit,
        args.arrow_count,
        args.dpi,
        vtp_points,
        args.vtp_alpha,
    )
    snapshots = make_snapshot_figure(
        flow_files,
        out_dir,
        time_labels,
        args.point_limit,
        args.arrow_count,
        args.dpi,
        args.snapshots,
        vtp_points,
        args.vtp_alpha,
    )
    hemo_grid = make_hemodynamic_grid(
        case_dir,
        selected,
        out_dir,
        time_labels.get(selected.stem, ""),
        args.point_limit,
        args.dpi,
        vtp_points,
        args.vtp_alpha,
        args.hemo_cmap,
        args.hemo_zoom,
        args.legacy_wss_scale,
        args.angiogram_path.resolve() if args.angiogram_path else None,
    )
    gif = (
        None
        if args.no_gif
        else maybe_make_gif(
            flow_files, out_dir, args.point_limit, args.arrow_count, max(120, args.dpi // 2)
        )
    )

    print(f"Case directory: {case_dir}")
    print(f"Flow timesteps found: {len(flow_files)}")
    print(f"VTP context: {vtp_path if vtp_points is not None else 'not available'}")
    print(f"Single-timestep figure: {single}")
    print(f"Snapshot figure: {snapshots}")
    print(f"Six-field hemodynamic figure: {hemo_grid}")
    if gif is not None:
        print(f"Animation: {gif}")


if __name__ == "__main__":
    main()
