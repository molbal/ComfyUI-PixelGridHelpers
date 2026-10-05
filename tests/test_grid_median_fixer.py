import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("pixelgrid_nodes", MODULE_PATH)
NODES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODES)


class GridMedianFixerTests(unittest.TestCase):
    def setUp(self):
        self.node = NODES.GridMedianFixer()

    def test_selects_nonzero_phase_and_removes_single_pixel_noise(self):
        grid_size = 3
        colors = torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]],
                    [[0.4, 0.2, 0.1], [0.5, 0.4, 0.2], [0.6, 0.6, 0.3]],
                    [[0.7, 0.1, 0.4], [0.8, 0.3, 0.5], [0.9, 0.5, 0.6]],
                ]
            ],
            dtype=torch.float32,
        )
        image = torch.ones((1, 12, 12, 3), dtype=torch.float32)
        for row in range(3):
            for column in range(3):
                top = 1 + row * grid_size
                left = 2 + column * grid_size
                image[:, top:top + grid_size, left:left + grid_size, :] = colors[
                    :, row:row + 1, column:column + 1, :
                ]

        image[0, 2, 3, :] = 0.0

        downscaled, upscaled = self.node.process_grid(image, grid_size)

        self.assertTrue(torch.equal(downscaled, colors))
        self.assertEqual(downscaled.shape, (1, 3, 3, 3))
        self.assertEqual(upscaled.shape, (1, 9, 9, 3))
        self.assertTrue(
            torch.equal(
                upscaled,
                colors.repeat_interleave(grid_size, dim=1).repeat_interleave(
                    grid_size, dim=2
                ),
            )
        )

    def test_breaks_equal_scores_in_favor_of_top_left_phase(self):
        image = torch.full((1, 13, 14, 3), 0.4, dtype=torch.float32)

        downscaled, upscaled = self.node.process_grid(image, grid_size=3)

        self.assertEqual(downscaled.shape, (1, 4, 4, 3))
        self.assertEqual(upscaled.shape, (1, 12, 12, 3))
        self.assertTrue(torch.all(downscaled == 0.4))
        self.assertTrue(torch.all(upscaled == 0.4))

    def test_grid_size_one_preserves_image_and_batch(self):
        image = torch.rand((2, 5, 7, 3), dtype=torch.float32)

        downscaled, upscaled = self.node.process_grid(image, grid_size=1)

        self.assertTrue(torch.equal(downscaled, image))
        self.assertTrue(torch.equal(upscaled, image))

    def test_image_smaller_than_grid_returns_empty_complete_block_dimensions(self):
        image = torch.rand((1, 2, 4, 3), dtype=torch.float32)

        downscaled, upscaled = self.node.process_grid(image, grid_size=4)

        self.assertEqual(downscaled.shape, (1, 0, 1, 3))
        self.assertEqual(upscaled.shape, (1, 0, 4, 3))

    def test_rejects_nonpositive_grid_size(self):
        image = torch.rand((1, 4, 4, 3), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "grid_size must be at least 1"):
            self.node.process_grid(image, grid_size=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_keeps_cuda_images_on_device(self):
        image = torch.rand((1, 8, 8, 3), dtype=torch.float32, device="cuda")

        downscaled, upscaled = self.node.process_grid(image, grid_size=2)

        self.assertEqual(downscaled.device.type, "cuda")
        self.assertEqual(upscaled.device.type, "cuda")
        self.assertEqual(downscaled.dtype, image.dtype)
        self.assertEqual(upscaled.dtype, image.dtype)


if __name__ == "__main__":
    unittest.main()
