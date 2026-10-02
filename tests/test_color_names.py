import importlib.util
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "color_names.py"
SPEC = importlib.util.spec_from_file_location("pixelgrid_color_names", MODULE_PATH)
COLOR_NAMES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLOR_NAMES)

VECTORS = (
    ("#000000", "Black", "#000000", 0),
    ("#FFFFFF", "White", "#FFFFFF", 0),
    ("#800080", "Purple", "#800080", 0),
    ("#808080", "Gray", "#808080", 0),
    ("#FF00FF", "Fuchsia", "#FF00FF", 0),
    ("#2563EB", "Royal Blue Shade", "#2563EB", 0),
    ("#36C", "Clear Blue", "#4566D3", 0.0210164591926121),
    ("#123456", "Muted Deep Blue", "#283252", 0.025634324770997356),
    ("#2F6FE4", "Royal Blue", "#4169E1", 0.01874073428901052),
    ("#FF5733", "Tomato", "#FF6347", 0.022021333004088272),
    ("#522828", "Muted Deep Red", "#522828", 0),
    ("#F9A9BE", "Vivid Pale Rose", "#F9A9BE", 0),
)


class ColorNameTests(unittest.TestCase):
    def setUp(self):
        self.node = COLOR_NAMES.PixelGrid_HexToNames()

    def test_source_vectors(self):
        for value, name, reference, distance in VECTORS:
            with self.subTest(value=value):
                self.assertEqual(COLOR_NAMES.color_name(value), name)
                target = COLOR_NAMES._hex_to_oklab(COLOR_NAMES._normalize_hex(value))
                expected = COLOR_NAMES._hex_to_oklab(reference)
                self.assertAlmostEqual(math.dist(target, expected), distance, delta=1e-12)
        palette = ", ".join(value for value, _, _, _ in VECTORS)
        self.assertEqual(
            self.node.name_colors(palette),
            (", ".join(name for _, name, _, _ in VECTORS),),
        )

    def test_all_library_colors_return_their_retained_exact_names(self):
        path = MODULE_PATH.parent / "assets" / "color-names.json"
        with path.open(encoding="utf-8") as file:
            entries = json.load(file)
        self.assertEqual(len(entries), 686)
        self.assertEqual(len({entry["hex"] for entry in entries}), 686)
        for entry in entries:
            with self.subTest(hex_color=entry["hex"]):
                self.assertEqual(COLOR_NAMES.color_name(entry["hex"]), entry["name"])

    def test_normalization(self):
        for value, expected in (
            (" #36c ", "#3366CC"),
            ("fff", "#FFFFFF"),
            ("800080", "#800080"),
            ("#aAbBcC", "#AABBCC"),
        ):
            with self.subTest(value=value):
                self.assertEqual(COLOR_NAMES._normalize_hex(value), expected)
        self.assertEqual(
            self.node.name_colors("  f00,\n00ff00, #00F  "),
            ("Red, Lime, Blue",),
        )

    def test_distinct_names_preserve_first_occurrence_order(self):
        self.assertEqual(
            self.node.name_colors("#FFFFFF, #000000, #FFFFFF, #FF00FF, #D946EF"),
            ("White, Black, Fuchsia",),
        )

    def test_normalized_and_nearest_matches_are_deduplicated_by_name(self):
        self.assertEqual(
            self.node.name_colors("#fff, FFFFFF, #123456, #283252, #FFF"),
            ("White, Muted Deep Blue",),
        )

    def test_empty_entries_follow_existing_palette_node_conventions(self):
        self.assertEqual(self.node.name_colors(" , \n, "), ("",))
        self.assertEqual(self.node.name_colors(", #000000,, #FFFFFF, "), ("Black, White",))

    def test_invalid_colors_fail_without_a_fallback(self):
        for value in (
            "#", "12", "1234", "12345", "1234567", "#11223344",
            "##fff", "f f f", "GG0000", "rgb(0,0,0)", "#fff\n000",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Invalid HEX color"):
                    COLOR_NAMES.color_name(value)
                with self.assertRaisesRegex(ValueError, "Invalid HEX color"):
                    self.node.name_colors(f"#000000, {value}, #FFFFFF")
        with self.assertRaisesRegex(ValueError, "Invalid HEX color"):
            COLOR_NAMES.color_name("")

    def test_equal_distances_keep_dataset_order(self):
        target = COLOR_NAMES._hex_to_oklab("#123456")
        entries = (
            ("First", "#000000", target),
            ("Second", "#FFFFFF", target),
        )
        with patch.object(COLOR_NAMES, "_COLORS", entries):
            self.assertEqual(COLOR_NAMES.color_name("#123456"), "First")

    def test_exact_names_take_precedence_over_nearest_search(self):
        with patch.object(COLOR_NAMES, "_hex_to_oklab", side_effect=AssertionError):
            self.assertEqual(COLOR_NAMES.color_name("#ff00ff"), "Fuchsia")

    def test_node_contract(self):
        self.assertEqual(self.node.INPUT_TYPES()["required"]["hex_palette"][0], "STRING")
        self.assertEqual(self.node.RETURN_TYPES, ("STRING",))
        self.assertEqual(self.node.RETURN_NAMES, ("color_names",))
        self.assertEqual(self.node.CATEGORY, "Pixel Grid Helpers")
        self.assertEqual(getattr(self.node, self.node.FUNCTION)("#000"), ("Black",))


if __name__ == "__main__":
    unittest.main()
