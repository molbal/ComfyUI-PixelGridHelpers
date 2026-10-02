import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "autofixer.py"
SPEC = importlib.util.spec_from_file_location("pixelgrid_autofixer", MODULE_PATH)
AUTOFIXER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOFIXER)


def fixture(accent, points):
    image = torch.empty((1, 16, 16, 3), dtype=torch.float32)
    image[:] = torch.tensor([205, 150, 125]) / 255
    image[:, 0:5] = torch.tensor([86, 44, 24]) / 255
    image[:, 12:] = torch.tensor([62, 31, 20]) / 255
    image[:, 6:10, 5:11] = torch.tensor([232, 190, 170]) / 255
    for y, x in points:
        image[0, y, x] = torch.tensor(accent) / 255
    return image


def output_colors(image):
    values = (image * 255).round().to(torch.uint8).reshape(-1, 3)
    return {tuple(color) for color in torch.unique(values, dim=0).tolist()}


def palette_colors(value):
    return {
        tuple(int(item[index : index + 2], 16) for index in (1, 3, 5))
        for item in value.split(", ")
    }


class AutofixerAccentTests(unittest.TestCase):
    def setUp(self):
        self.node = AUTOFIXER.PixelGrid_Autofixer()

    def test_preserves_small_vibrant_iris_at_four_colors(self):
        green = (20, 235, 65)
        image = fixture(green, [(7, 7), (7, 8), (8, 7)])

        result, palette, _ = self.node.autofix(
            image, pixel_size=1, colors="4 colors"
        )

        self.assertIn(green, output_colors(result))
        self.assertLessEqual(len(palette.split(", ")), 4)

    def test_preserves_small_silver_jewelry_at_four_colors(self):
        silver = (205, 225, 235)
        image = fixture(silver, [(8, 3), (9, 3), (10, 3)])

        result, palette, _ = self.node.autofix(
            image, pixel_size=1, colors="4 colors"
        )

        self.assertIn(silver, output_colors(result))
        self.assertLessEqual(len(palette.split(", ")), 4)

    def test_preserves_vibrant_eyes_and_silver_jewelry_together(self):
        green = (20, 235, 65)
        silver = (205, 225, 235)
        image = fixture(green, [(7, 7), (7, 8), (8, 7)])
        for y, x in [(8, 3), (9, 3), (10, 3)]:
            image[0, y, x] = torch.tensor(silver) / 255

        result, palette, _ = self.node.autofix(
            image, pixel_size=1, colors="8 colors"
        )

        colors = output_colors(result)
        self.assertIn(green, colors)
        self.assertIn(silver, colors)
        self.assertLessEqual(len(palette.split(", ")), 8)

    def test_preserves_a_three_shade_iris_ramp_at_twelve_colors(self):
        shades = ((93, 189, 64), (70, 130, 67), (33, 91, 25))
        image = torch.empty((1, 24, 24, 3), dtype=torch.float32)
        background = [
            (55 + index * 9, 30 + index * 6, 20 + index * 5)
            for index in range(16)
        ]
        for tile_y in range(4):
            for tile_x in range(4):
                color = background[tile_y * 4 + tile_x]
                image[
                    :,
                    tile_y * 6 : (tile_y + 1) * 6,
                    tile_x * 6 : (tile_x + 1) * 6,
                ] = torch.tensor(color) / 255
        for color, points in zip(
            shades,
            (
                [(10, 10), (10, 11)],
                [(11, 10), (11, 11)],
                [(12, 10), (12, 11)],
            ),
        ):
            for y, x in points:
                image[0, y, x] = torch.tensor(color) / 255
        source = (image * 255).round().to(torch.uint8)
        provisional = {
            tuple(color)
            for color in AUTOFIXER._source_palette(source, 12).tolist()
        }

        result, palette, _ = self.node.autofix(
            image, pixel_size=1, colors="12 colors - recommended"
        )

        self.assertNotIn(shades[1], provisional)
        self.assertNotIn(shades[2], provisional)
        colors = output_colors(result)
        for shade in shades:
            self.assertIn(shade, colors)
        self.assertLessEqual(len(palette.split(", ")), 12)

    def test_repeated_single_pixel_eye_accents_are_coherent(self):
        green = (20, 235, 65)
        image = fixture(green, [(7, 7), (7, 10)])

        result, _, _ = self.node.autofix(
            image, pixel_size=1, colors="4 colors"
        )

        self.assertIn(green, output_colors(result))

    def test_isolated_chromatic_noise_does_not_reserve_a_slot(self):
        image = fixture((255, 0, 255), [(8, 7)])
        source = (image * 255).round().to(torch.uint8)

        colors, assignments = AUTOFIXER._accent_plan(source, 4)

        self.assertEqual(colors.numel(), 0)
        self.assertFalse(bool((assignments >= 0).any()))

    def test_existing_broad_palette_color_does_not_waste_an_accent_slot(self):
        image = fixture((0, 0, 0), [(7, 7), (7, 8), (8, 7)])
        source = (image * 255).round().to(torch.uint8)
        reference = AUTOFIXER._source_palette(source, 4)

        colors, _ = AUTOFIXER._accent_plan(source, 4, reference)

        self.assertNotIn((0, 0, 0), {tuple(color) for color in colors.tolist()})

    def test_long_one_pixel_split_line_is_not_an_accent(self):
        gray = (145, 145, 145)
        image = fixture(gray, [(y, 7) for y in range(4, 10)])
        source = (image * 255).round().to(torch.uint8)

        candidates = AUTOFIXER._analyze_accent_frame(source[0])

        self.assertNotIn(gray, {candidate["color"] for candidate in candidates})

    def test_cleanup_never_removes_a_protected_accent(self):
        image = torch.full((5, 5, 3), 100, dtype=torch.uint8)
        image[2, 2] = torch.tensor([104, 100, 100], dtype=torch.uint8)
        protection = torch.zeros((5, 5), dtype=torch.bool)
        protection[2, 2] = True

        unprotected = AUTOFIXER._clean_small_islands(image)
        protected = AUTOFIXER._clean_small_islands(image, protection)

        self.assertEqual(tuple(unprotected[2, 2].tolist()), (100, 100, 100))
        self.assertEqual(tuple(protected[2, 2].tolist()), (104, 100, 100))

    def test_two_color_mode_uses_the_original_broad_palette_path(self):
        image = fixture((20, 235, 65), [(7, 7), (7, 8), (8, 7)])
        source = (image * 255).round().to(torch.uint8)
        palette = AUTOFIXER._source_palette(source, 2)
        expected = torch.stack(
            [
                AUTOFIXER._clean_small_islands(frame)
                for frame in AUTOFIXER._map_palette(source, palette)
            ]
        ).to(torch.float32) / 255

        result, output_palette, _ = self.node.autofix(
            image, pixel_size=1, colors="2 colors - two-tone"
        )

        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(output_palette, AUTOFIXER._palette_string(palette))

    def test_custom_palette_remains_authoritative_and_protects_accents(self):
        green = (0, 255, 0)
        image = fixture((20, 235, 65), [(7, 7), (7, 8), (8, 7)])

        result, palette, _ = self.node.autofix(
            image,
            pixel_size=1,
            colors="4 colors",
            palette="#000000,#8b4513,#f2b9a0,#00ff00",
        )

        self.assertEqual(palette, "#000000, #8b4513, #f2b9a0, #00ff00")
        self.assertIn(green, output_colors(result))

    def test_every_generated_palette_respects_its_hard_limit(self):
        values = torch.arange(256, dtype=torch.float32).reshape(1, 16, 16, 1) / 255
        image = torch.cat(
            (values, torch.flip(values, [1]), torch.flip(values, [2])), dim=-1
        )
        options = self.node.INPUT_TYPES()["optional"]["colors"][0]

        for option in options:
            with self.subTest(option=option):
                result, palette, _ = self.node.autofix(
                    image, pixel_size=1, colors=option
                )
                limit = int(option.split()[0])
                self.assertLessEqual(len(palette.split(", ")), limit)
                self.assertLessEqual(len(output_colors(result)), limit)
                self.assertLessEqual(palette_colors(palette), output_colors(result))

    def test_accent_near_protection_limit_is_not_starved(self):
        image = torch.empty((1, 60, 60, 3), dtype=torch.float32)
        image[:] = torch.tensor([205, 150, 125]) / 255
        image[:, :, :20] = torch.tensor([86, 44, 24]) / 255
        image[:, :, 40:] = torch.tensor([62, 31, 20]) / 255
        green = (20, 235, 65)
        image[:, 25:34, 25:34] = torch.tensor(green) / 255

        result, palette, _ = self.node.autofix(
            image, pixel_size=1, colors="4 colors"
        )

        self.assertIn(green, output_colors(result))
        self.assertIn(green, palette_colors(palette))

    def test_all_sampling_modes_preserve_shape_and_batch(self):
        image = torch.rand((2, 13, 17, 3), dtype=torch.float32)
        modes = self.node.INPUT_TYPES()["optional"]["sampling"][0]

        for mode in modes:
            with self.subTest(mode=mode):
                result, _, used = self.node.autofix(
                    image,
                    pixel_size=4,
                    sampling=mode,
                    colors="8 colors",
                )
                self.assertEqual(result.shape, image.shape)
                self.assertEqual(result.dtype, image.dtype)
                self.assertEqual(used, 4)

    def test_non_divisible_dimensions_restore_exact_pixel_blocks(self):
        levels = torch.tensor([0, 80, 160, 255], dtype=torch.float32) / 255
        row = torch.tensor(
            [0, 0, 0, 0, 80, 80, 0, 160, 160, 255], dtype=torch.float32
        ) / 255
        image = row.reshape(1, 1, 10, 1).repeat(1, 3, 1, 3)
        palette = ",".join(
            f"#{value:02x}{value:02x}{value:02x}"
            for value in (0, 80, 160, 255)
        )

        result, _, _ = self.node.autofix(
            image,
            pixel_size=3,
            palette=palette,
            sampling="Exact pixels - existing pixel art",
        )

        expected = levels.repeat_interleave(3)[:10]
        self.assertTrue(torch.equal(result[0, 0, :, 0], expected))

    def test_batch_palette_and_output_are_deterministic(self):
        generator = torch.Generator().manual_seed(42)
        image = torch.rand((2, 24, 20, 3), generator=generator)

        first = self.node.autofix(
            image, pixel_size=2, colors="12 colors - recommended"
        )
        second = self.node.autofix(
            image, pixel_size=2, colors="12 colors - recommended"
        )

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1:], second[1:])


if __name__ == "__main__":
    unittest.main()
