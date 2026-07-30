# Version 14 source snapshot
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rupture_status import (
    filter_predictions_to_known_status,
    normalize_rupture_status,
    rupture_target,
)


class RuptureStatusTests(unittest.TestCase):
    def test_only_explicit_known_values_become_targets(self):
        self.assertEqual(normalize_rupture_status(" Ruptured "), "ruptured")
        self.assertEqual(rupture_target("unruptured"), 0)
        self.assertIsNone(rupture_target("unknown"))
        self.assertIsNone(rupture_target(""))
        self.assertIsNone(rupture_target(float("nan")))

    def test_prediction_filter_removes_blank_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.csv"
            pd.DataFrame(
                [
                    {
                        "dataset": "known",
                        "vesselFileID": "known",
                        "cutToShow": "cut1",
                        "status": "ruptured",
                    },
                    {
                        "dataset": "blank",
                        "vesselFileID": "blank",
                        "cutToShow": "cut1",
                        "status": "",
                    },
                ]
            ).to_csv(metadata_path, index=False)
            predictions = pd.DataFrame(
                [
                    {
                        "filepath": "root/known_cut1/hemodynamics_aggregate.csv",
                        "label": 1,
                        "prob": 0.8,
                        "pred": 1,
                    },
                    {
                        "filepath": "root/blank_cut1/hemodynamics_aggregate.csv",
                        "label": 0,
                        "prob": 0.2,
                        "pred": 0,
                    },
                ]
            )
            kept, excluded = filter_predictions_to_known_status(predictions, metadata_path)
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded.iloc[0]["case_name"], "blank_cut1")
            self.assertEqual(kept.iloc[0]["pred"], 1)


if __name__ == "__main__":
    unittest.main()
