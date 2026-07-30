# Version 3 source snapshot
import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, Response, jsonify, request
from plotly.subplots import make_subplots

# Minimal Flask app that shows a Plotly cone plot (velocity) for a random CSV
app = Flask(__name__)

# Folders to search
DATA_DIRS = [
    "predictions/batch_results",
    "predictions/optimized_pinn",
]
PATTERN = "*.csv"

# Optional CSV path provided via CLI; when set, that file is visualized instead of a random pick
SELECTED_CSV = None


def list_csvs():
    files = []
    for d in DATA_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, PATTERN))))
    return files


def pick_random_csv():
    global SELECTED_CSV
    if SELECTED_CSV:
        if os.path.exists(SELECTED_CSV):
            return SELECTED_CSV
        else:
            raise FileNotFoundError(f"Requested CSV not found: {SELECTED_CSV}")
    # Prefer: individual CSVs in batch_results and folder-cases in optimized_pinn
    batch_dir = os.path.join("predictions", "batch_results")
    opt_dir = os.path.join("predictions", "optimized_pinn")

    batch_files = (
        sorted(glob.glob(os.path.join(batch_dir, PATTERN))) if os.path.isdir(batch_dir) else []
    )
    opt_subdirs = []
    if os.path.isdir(opt_dir):
        for name in sorted(os.listdir(opt_dir)):
            full = os.path.join(opt_dir, name)
            if os.path.isdir(full):
                opt_subdirs.append(full)

    candidates = batch_files + opt_subdirs
    if not candidates:
        # fallback to searching all DATA_DIRS for csvs
        files = list_csvs()
        if not files:
            raise FileNotFoundError(
                f"No CSV files or optimized_pinn subfolders found in {DATA_DIRS}"
            )
        return random.choice(files)

    return random.choice(candidates)


def build_velocity_figure(csv_path: str, color_param: str = None, box_params: list | None = None):
    # Handle either a single CSV or a folder of CSV timesteps (average)
    if os.path.isdir(csv_path):
        # Prefer a 'timesteps' subfolder with files like flow_t*.csv (common optimized_pinn layout)
        timedir = None
        candidates = []
        # check common subpaths
        for sub in ("timesteps", "time_steps", "time-series", "frames"):
            p = os.path.join(csv_path, sub)
            if os.path.isdir(p):
                timedir = p
                break
        if timedir is None:
            # if the directory itself contains CSVs, use them
            candidates = sorted(glob.glob(os.path.join(csv_path, "*.csv")))
        else:
            # prefer files matching flow_t*.csv, else all CSVs
            candidates = sorted(glob.glob(os.path.join(timedir, "flow_t*.csv")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(timedir, "*.csv")))

        if not candidates:
            raise FileNotFoundError(
                f"No timestep CSV files found in folder {csv_path} (looked in {timedir or csv_path})"
            )

        # Read first file to capture coordinates and shape
        df0 = pd.read_csv(candidates[0])
        colmap0 = {c.lower(): c for c in df0.columns}
        needed = ("x", "y", "z", "u", "v", "w")
        if not all(k in colmap0 for k in needed):
            raise ValueError(
                f"CSV {candidates[0]} missing required columns (x,y,z,u,v,w). Found: {list(df0.columns)}"
            )

        x = df0[colmap0["x"]].to_numpy()
        y = df0[colmap0["y"]].to_numpy()
        z = df0[colmap0["z"]].to_numpy()

        vecs = []
        # collect additional scalar columns across timesteps
        scalar_names = None
        scalar_vals = {}
        for f in candidates:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            colmap = {c.lower(): c for c in df.columns}
            if not all(k in colmap for k in needed):
                continue
            u = df[colmap["u"]].to_numpy()
            v = df[colmap["v"]].to_numpy()
            w = df[colmap["w"]].to_numpy()
            if u.shape[0] != x.shape[0]:
                # skip files with mismatched point counts
                continue
            vecs.append(np.stack([u, v, w], axis=1))

            # handle extra scalar columns (anything not x,y,z,u,v,w)
            extras = [c for c in df.columns if c.lower() not in ("x", "y", "z", "u", "v", "w")]
            if scalar_names is None:
                scalar_names = [c.lower() for c in extras]
                for s in scalar_names:
                    scalar_vals[s] = []
            # collect values for named scalars (if present)
            for s in scalar_names or []:
                col = colmap.get(s)
                if col and col in df.columns:
                    scalar_vals[s].append(df[col].to_numpy())

        if not vecs:
            raise ValueError(f"No valid timesteps with consistent u/v/w found in {csv_path}")

        stacked = np.stack(vecs, axis=0)  # (n_files, n_points, 3)
        avg_vec = np.mean(stacked, axis=0)  # (n_points, 3)
        u = avg_vec[:, 0]
        v = avg_vec[:, 1]
        w = avg_vec[:, 2]
        mag = np.linalg.norm(avg_vec, axis=1)
        title = os.path.basename(csv_path.rstrip(os.sep))
        # average scalar fields across timesteps (if any)
        avg_scalars = {}
        if scalar_names:
            for s in scalar_names:
                lists = scalar_vals.get(s, [])
                if not lists:
                    continue
                stacked_s = np.vstack(lists)
                avg_s = np.nanmean(stacked_s, axis=0)
                if avg_s.shape[0] == x.shape[0]:
                    avg_scalars[s] = avg_s
    else:
        # Read with pandas for robustness
        df = pd.read_csv(csv_path)
        colmap = {c.lower(): c for c in df.columns}
        needed = ("x", "y", "z", "u", "v", "w")
        if not all(k in colmap for k in needed):
            raise ValueError(
                f"CSV {csv_path} missing required columns (x,y,z,u,v,w). Found: {list(df.columns)}"
            )

        x = df[colmap["x"]].to_numpy()
        y = df[colmap["y"]].to_numpy()
        z = df[colmap["z"]].to_numpy()
        u = df[colmap["u"]].to_numpy()
        v = df[colmap["v"]].to_numpy()
        w = df[colmap["w"]].to_numpy()

        mag = np.linalg.norm(np.stack([u, v, w], axis=1), axis=1)
        title = os.path.basename(csv_path)
        # collect any additional scalars in this single CSV
        avg_scalars = {}
        extras = [c for c in df.columns if c.lower() not in ("x", "y", "z", "u", "v", "w")]
        for c in extras:
            arr = df[c].to_numpy()
            if arr.shape[0] == x.shape[0]:
                avg_scalars[c.lower()] = arr

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "scene"}]])

    # Single-parameter box plot logic: choose one parameter to plot
    # Priority: explicit box_params[0] -> color_param -> tawss -> velocity_mag
    fig.add_trace(go.Box(y=mag, name="velocity_mag"), row=1, col=1)
    primary_color_scalar = None
    selected_box = None
    if box_params:
        # use only the first requested box param
        candidate = box_params[0].lower()
        if candidate in avg_scalars:
            selected_box = candidate
    if selected_box is None and color_param:
        cp = color_param.lower()
        if cp in avg_scalars:
            selected_box = cp
    if selected_box is None and "tawss" in avg_scalars:
        selected_box = "tawss"
    # attach only the selected box (if any)
    if selected_box:
        fig.add_trace(go.Box(y=avg_scalars[selected_box], name=selected_box), row=1, col=1)
        primary_color_scalar = selected_box

    # Cone plot for vectors
    cone = go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        sizemode="absolute",
        sizeref=max(1.0, np.nanmax(mag)) * 2,
        colorscale="Viridis",
        colorbar=dict(title="|v|"),
        showscale=True,
        anchor="tail",
        hoverinfo="skip",
    )
    fig.add_trace(cone, row=1, col=2)

    # overlay scatter colored by selected color_param (or primary_color_scalar or magnitude)
    color_vals = None
    if color_param:
        key = color_param.lower()
        if key in avg_scalars:
            color_vals = avg_scalars[key]
        elif key == "velocity_mag":
            color_vals = mag
    if color_vals is None:
        if primary_color_scalar and primary_color_scalar in avg_scalars:
            color_vals = avg_scalars[primary_color_scalar]
        else:
            color_vals = mag
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                color=color_vals,
                colorscale="Viridis",
                size=3,
                colorbar=dict(title=(color_param or primary_color_scalar or "|v|")),
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title=title, scene=dict(aspectmode="data"))

    # hide grid lines / axis visuals for a cleaner view
    fig.update_scenes(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        zaxis_showgrid=False,
        xaxis_showaxeslabels=False,
        yaxis_showaxeslabels=False,
        zaxis_showaxeslabels=False,
        xaxis_showbackground=False,
        yaxis_showbackground=False,
        zaxis_showbackground=False,
        xaxis_showline=False,
        yaxis_showline=False,
        zaxis_showline=False,
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        zaxis_showticklabels=False,
        xaxis_nticks=0,
        yaxis_nticks=0,
        zaxis_nticks=0,
        aspectmode="data",
    )

    return fig


@app.route("/")
def index():
    # query args: csv (path), color (param name), boxes (comma-separated list)
    try:
        csv_arg = request.args.get("csv")
        color_arg = request.args.get("color")
        boxes_arg = request.args.get("boxes")
        boxes = [b.strip() for b in boxes_arg.split(",")] if boxes_arg else None

        if csv_arg:
            csv_path = csv_arg
        else:
            csv_path = pick_random_csv()

        fig = build_velocity_figure(csv_path, color_param=color_arg, box_params=boxes)
        html = pio.to_html(fig, full_html=True)
        return Response(html, mimetype="text/html")
    except Exception as e:
        return Response(f"Error building plot: {e}", mimetype="text/plain")


@app.route("/scalars")
def scalars():
    # returns JSON list of available scalar names for a given csv/folder (use ?csv=...)
    csv_arg = request.args.get("csv")
    try:
        if not csv_arg:
            csv_path = pick_random_csv()
        else:
            csv_path = csv_arg
        # build figure but only to extract avg_scalars without plotting; reuse logic
        # call build_velocity_figure but capture avg_scalars by invoking internal path: read dir or file
        if os.path.isdir(csv_path):
            timedir = None
            for sub in ("timesteps", "time_steps", "time-series", "frames"):
                p = os.path.join(csv_path, sub)
                if os.path.isdir(p):
                    timedir = p
                    break
            candidates = sorted(glob.glob(os.path.join(timedir or csv_path, "flow_t*.csv")))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(timedir or csv_path, "*.csv")))
            scalars = set()
            for f in candidates:
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
                for c in df.columns:
                    if c.lower() not in ("x", "y", "z", "u", "v", "w"):
                        scalars.add(c.lower())
            return jsonify(sorted(list(scalars)))
        else:
            df = pd.read_csv(csv_path)
            scalars = [
                c.lower() for c in df.columns if c.lower() not in ("x", "y", "z", "u", "v", "w")
            ]
            return jsonify(sorted(scalars))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve velocity visualization")
    parser.add_argument(
        "--csv-file", "-c", help="Path to CSV file to visualize (overrides random pick)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8050, help="Port to run the Flask server on"
    )
    args = parser.parse_args()
    if args.csv_file:
        SELECTED_CSV = args.csv_file

    # Run Flask app
    app.run(host="127.0.0.1", port=args.port, debug=True)
