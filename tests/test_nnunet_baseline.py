from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.nnunet_common import parse_nnunet_validation_summary, write_splits_final
from scripts.prepare_nnunet_dataset import _build_case_day_volumes
from src.rle import rle_encode


class NNUNetBaselineTests(unittest.TestCase):
    def test_write_splits_final_matches_nnunet_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_splits_final(
                Path(tmp_dir),
                train_ids=["case1_day0", "case2_day0"],
                val_ids=["case3_day0"],
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload,
            [{"train": ["case1_day0", "case2_day0"], "val": ["case3_day0"]}],
        )

    def test_parse_nnunet_validation_summary_extracts_dice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "foreground_mean": {"Dice": 0.81},
                        "mean": {
                            "1": {"Dice": 0.7},
                            "2": {"Dice": 0.8},
                            "3": {"Dice": 0.93},
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = parse_nnunet_validation_summary(
                summary_path,
                class_names=["large_bowel", "small_bowel", "stomach"],
                label_values=[1, 2, 3],
            )
        self.assertAlmostEqual(metrics["official_mean_dice"], 0.81)
        self.assertAlmostEqual(metrics["official_dice_large_bowel"], 0.7)
        self.assertAlmostEqual(metrics["official_dice_small_bowel"], 0.8)
        self.assertAlmostEqual(metrics["official_dice_stomach"], 0.93)

    def test_label_collapse_uses_fixed_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.zeros((4, 4), dtype=np.uint16)
            slice_path = Path(tmp_dir) / "slice_0001_4_4_1.0_1.0.png"
            ok = cv2.imwrite(str(slice_path), image)
            self.assertTrue(ok)

            large_bowel = np.zeros((4, 4), dtype=np.uint8)
            stomach = np.zeros((4, 4), dtype=np.uint8)
            large_bowel[1, 1] = 1
            stomach[1, 1] = 1
            stomach[2, 2] = 1

            rle_index = {
                ("case1_day0", 1): {
                    "large_bowel": rle_encode(large_bowel),
                    "small_bowel": "",
                    "stomach": rle_encode(stomach),
                }
            }
            _, label_volume, _, stats = _build_case_day_volumes(
                case_day="case1_day0",
                slice_paths=[str(slice_path)],
                rle_index=rle_index,
                label_priority=["large_bowel", "small_bowel", "stomach"],
                spacing_z_override=None,
            )

        self.assertEqual(int(label_volume[0, 1, 1]), 3)
        self.assertEqual(int(label_volume[0, 2, 2]), 3)
        self.assertEqual(int(stats["overlap_slices"]), 1)
        self.assertEqual(int(stats["overwritten_voxels"]), 1)


if __name__ == "__main__":
    unittest.main()
