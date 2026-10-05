import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("pixelgrid_nodes", MODULE_PATH)
NODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODES)


class CreateSolidColorImageTests(unittest.TestCase):
    def setUp(self):
        self.node = NODES.PixelGrid_CreateSolidColorImage()

    def test_node_contract(self):
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(self.node.RETURN_NAMES, ("image",))
        self.assertEqual(self.node.CATEGORY, "Pixel Grid Helpers")
        self.assertEqual(self.node.FUNCTION, "create_image")
        self.assertEqual(
            set(self.node.INPUT_TYPES()["required"]),
            {"width", "height", "hex_code"},
        )

    def test_creates_rgb_image_with_requested_dimensions_and_color(self):
        for hex_code, expected in (
            ("#123456", [0x12 / 255, 0x34 / 255, 0x56 / 255]),
            (" f0A ", [1.0, 0.0, 170 / 255]),
        ):
            with self.subTest(hex_code=hex_code):
                image, = self.node.create_image(3, 2, hex_code)
                self.assertEqual(image.shape, (1, 2, 3, 3))
                self.assertEqual(image.dtype, torch.float32)
                self.assertTrue(
                    torch.allclose(
                        image,
                        torch.tensor(expected, dtype=torch.float32).view(1, 1, 1, 3).expand_as(image),
                    )
                )

    def test_invalid_hex_colors_raise_a_clear_error(self):
        for hex_code in ("", "#", "12", "1234", "12345", "1234567", "GG0000"):
            with self.subTest(hex_code=hex_code):
                with self.assertRaisesRegex(ValueError, "Invalid HEX color"):
                    self.node.create_image(1, 1, hex_code)


if __name__ == "__main__":
    unittest.main()
