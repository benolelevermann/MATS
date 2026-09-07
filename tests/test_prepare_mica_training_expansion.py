from __future__ import annotations

import random
import unittest

from prepare_mica_c3_c10_training_expansion import (
    parse_plate_folder,
    random_fov_coordinates,
    stable_random,
)


class MicaTrainingExpansionTests(unittest.TestCase):
    def test_plate_folder_provenance_is_parsed(self) -> None:
        token, div, plate = parse_plate_folder(
            "2025_03_20_17_51_57--Niro_96well_eGFP "
            "250320_MCS24GFP_div07_plate2"
        )
        self.assertEqual(token, "250320_MCS24GFP_div07_plate2")
        self.assertEqual(div, 7)
        self.assertEqual(plate, 2)

    def test_random_fovs_are_reproducible_separated_and_in_bounds(self) -> None:
        first = random_fov_coordinates(
            20_322, 20_598, 2_048, 3, stable_random("example", 20260831)
        )
        second = random_fov_coordinates(
            20_322, 20_598, 2_048, 3, stable_random("example", 20260831)
        )
        self.assertEqual(first, second)
        for x, y in first:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + 2_048, 20_322)
            self.assertLessEqual(y + 2_048, 20_598)
        for index, (x, y) in enumerate(first):
            for other_x, other_y in first[index + 1 :]:
                self.assertGreaterEqual(
                    ((x - other_x) ** 2 + (y - other_y) ** 2) ** 0.5,
                    2_048,
                )

    def test_crop_larger_than_overview_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            random_fov_coordinates(100, 100, 200, 3, random.Random(1))


if __name__ == "__main__":
    unittest.main()
