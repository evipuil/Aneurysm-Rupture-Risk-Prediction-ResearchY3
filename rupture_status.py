# Version 14 source snapshot
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

KNOWN_RUPTURE_STATUSES = {"unruptured": 0, "ruptured": 1}


def normalize_rupture_status(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text if text in KNOWN_RUPTURE_STATUSES else None


def rupture_target(value) -> Optional[int]:
    status = normalize_rupture_status(value)
    return None if status is None else KNOWN_RUPTURE_STATUSES[status]


def build_case_status_index(metadata_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    metadata = pd.read_csv(metadata_path)
    key_to_idx: Dict[str, int] = {}
    for idx, row in metadata.iterrows():
        dataset = str(row.get("dataset", "")).strip()
        vessel_id = str(row.get("vesselFileID", "")).strip()
        cut_raw = row.get("cutToShow", "cut1")
        cut = str(cut_raw).strip() if pd.notna(cut_raw) else "cut1"
        for key in (dataset, vessel_id):
            if key:
                key_to_idx.setdefault(key, idx)
                key_to_idx.setdefault(f"{key}_{cut}", idx)
                key_to_idx.setdefault(f"{key}_cut1", idx)
    return metadata, key_to_idx


def match_case_index(case_name: str, key_to_idx: Dict[str, int]) -> Optional[int]:
    case_name = str(case_name).strip()
    if case_name in key_to_idx:
        return key_to_idx[case_name]
    base = re.sub(r"_cut\d+$", "", case_name)
    return key_to_idx.get(base)


def case_name_from_filepath(value: str) -> str:
    normalized = str(value).replace("\\", "/").rstrip("/")
    parts = normalized.split("/")
    if parts and parts[-1].lower().endswith(".csv"):
        parts = parts[:-1]
    return parts[-1] if parts else ""


def prediction_status(
    value: str, metadata: pd.DataFrame, key_to_idx: Dict[str, int]
) -> Optional[str]:
    case_name = case_name_from_filepath(value)
    idx = match_case_index(case_name, key_to_idx)
    if idx is None:
        return None
    return normalize_rupture_status(metadata.iloc[idx].get("status", None))


def filter_predictions_to_known_status(
    frame: pd.DataFrame,
    metadata_path: str | Path,
    filepath_column: str = "filepath",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if filepath_column not in frame.columns:
        raise ValueError(f"Prediction frame has no {filepath_column!r} column")
    metadata, key_to_idx = build_case_status_index(metadata_path)
    statuses = frame[filepath_column].map(
        lambda value: prediction_status(value, metadata, key_to_idx)
    )
    excluded = frame.loc[statuses.isna()].copy()
    excluded.insert(0, "case_name", excluded[filepath_column].map(case_name_from_filepath))
    kept = frame.loc[statuses.notna()].copy()
    expected_labels = (
        statuses.loc[statuses.notna()].map(KNOWN_RUPTURE_STATUSES).astype(int).to_numpy()
    )
    if "label" in kept.columns:
        observed = pd.to_numeric(kept["label"], errors="raise").astype(int).to_numpy()
        if not (observed == expected_labels).all():
            raise ValueError("Prediction labels disagree with known metadata rupture status")
    return kept.reset_index(drop=True), excluded.reset_index(drop=True)
