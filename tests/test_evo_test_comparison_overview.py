from __future__ import annotations

import unittest

from r_pipeline.build_evo_test_comparison_overview import normalize_source


class EvoTestComparisonOverviewTests(unittest.TestCase):
    def test_source_normalization_preserves_decimal_zoom_and_matches_tif(self) -> None:
        manual = (
            "260723_cc_M237_S24tdTomMFL26#02_div10_postROCKi_"
            "10x_zoom1.6_ERplate_E8_DMSO.tif"
        )
        automatic = (
            "260723_cc_M237_S24tdTomMFL26_02_div10_postROCKi_"
            "10x_zoom1.6_ERplate_E8_DMSO"
        )
        self.assertEqual(normalize_source(manual), normalize_source(automatic))


if __name__ == "__main__":
    unittest.main()
