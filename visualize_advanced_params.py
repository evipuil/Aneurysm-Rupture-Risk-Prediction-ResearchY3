# Version 1 source snapshot
import glob
import os

import numpy as np
import pyvista as pv

# CONFIG
DATA_DIR = "predictions/pinn_corrected"
DEFAULT_PATTERN = "*.csv"
POINT_SIZE = 5
BACKGROUND = "white"

# Scalars to cycle through (derived from CSV headers)
VECTOR_FIELD_NAME = "velocity"


# HELPERS
def load_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    cols = [c.strip() for c in header.split(",") if c.strip()]
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return cols, data


def list_csv_files():
    files = sorted(glob.glob(os.path.join(DATA_DIR, DEFAULT_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    return files


# MAIN
def main():
    files = list_csv_files()
    file_state = {"idx": 0}

    def load_file(idx: int):
        csv_path = files[idx]
        cols, data = load_csv(csv_path)
        return csv_path, cols, data

    csv_path, cols, data = load_file(file_state["idx"])

    xyz = data[:, 0:3]

    # Build scalar dict from all available columns (exclude coordinates)
    col_idx = {name: i for i, name in enumerate(cols)}
    scalar_cols = [c for c in cols if c not in ("x", "y", "z")]
    if not scalar_cols:
        raise ValueError("No scalar columns found in CSV header")

    scalars = {name: data[:, col_idx[name]] for name in scalar_cols}
    velocity = None
    if all(k in col_idx for k in ("u", "v", "w")):
        velocity = data[:, [col_idx["u"], col_idx["v"], col_idx["w"]]]

    mesh = pv.PolyData(xyz)
    for name, arr in scalars.items():
        mesh.point_data[name] = arr
    if velocity is not None:
        mesh.point_data[VECTOR_FIELD_NAME] = velocity
        mesh.point_data["velocity_mag"] = np.linalg.norm(velocity, axis=1)

    plotter = pv.Plotter()
    plotter.set_background(BACKGROUND)

    state = {"idx": 0}

    def show_scalar():
        sname = scalar_cols[state["idx"]]
        plotter.clear()

        if sname == VECTOR_FIELD_NAME:
            plotter.add_points(
                mesh,
                scalars="velocity_mag",
                render_points_as_spheres=True,
                point_size=POINT_SIZE,
                cmap="viridis",
            )
            plotter.add_text("Velocity magnitude |v|", position="upper_left", font_size=12)
        else:
            plotter.add_points(
                mesh,
                scalars=sname,
                render_points_as_spheres=True,
                point_size=POINT_SIZE,
                cmap="viridis",
            )
            plotter.add_text(f"Scalar: {sname}", position="upper_left", font_size=12)

        plotter.add_text(f"File: {os.path.basename(csv_path)}", position="lower_left", font_size=10)

        plotter.render()

    def next_scalar():
        state["idx"] = (state["idx"] + 1) % len(scalar_cols)
        show_scalar()

    def prev_scalar():
        state["idx"] = (state["idx"] - 1) % len(scalar_cols)
        show_scalar()

    def next_file():
        file_state["idx"] = (file_state["idx"] + 1) % len(files)
        reload_file()

    def prev_file():
        file_state["idx"] = (file_state["idx"] - 1) % len(files)
        reload_file()

    def reload_file():
        nonlocal csv_path, cols, data, mesh, scalars, velocity, scalar_cols, col_idx
        csv_path, cols, data = load_file(file_state["idx"])
        xyz = data[:, 0:3]
        col_idx = {name: i for i, name in enumerate(cols)}
        scalar_cols = [c for c in cols if c not in ("x", "y", "z")]
        scalars = {name: data[:, col_idx[name]] for name in scalar_cols}
        velocity = (
            data[:, [col_idx["u"], col_idx["v"], col_idx["w"]]]
            if all(k in col_idx for k in ("u", "v", "w"))
            else None
        )

        mesh = pv.PolyData(xyz)
        for name, arr in scalars.items():
            mesh.point_data[name] = arr
        if velocity is not None:
            mesh.point_data[VECTOR_FIELD_NAME] = velocity
            mesh.point_data["velocity_mag"] = np.linalg.norm(velocity, axis=1)

        state["idx"] = 0
        show_scalar()

    if velocity is not None and VECTOR_FIELD_NAME not in scalar_cols:
        scalar_cols.append(VECTOR_FIELD_NAME)

    plotter.add_key_event("n", next_scalar)
    plotter.add_key_event("p", prev_scalar)
    plotter.add_key_event("f", next_file)
    plotter.add_key_event("b", prev_file)

    show_scalar()
    plotter.show()


if __name__ == "__main__":
    main()
