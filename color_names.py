import json
import math
import re
from pathlib import Path


def _normalize_hex(value):
    match = re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", value.strip())
    if match is None:
        raise ValueError(f"Invalid HEX color {value!r}; use 3 or 6 hexadecimal digits.")
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(digit * 2 for digit in digits)
    return "#" + digits.upper()


def _hex_to_oklab(hex_color):
    channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    red, green, blue = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    l = (0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue) ** (1 / 3)
    m = (0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue) ** (1 / 3)
    s = (0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


with (Path(__file__).parent / "assets" / "color-names.json").open(encoding="utf-8") as file:
    _COLORS = tuple(
        (entry["name"], entry["hex"], _hex_to_oklab(entry["hex"]))
        for entry in json.load(file)
    )
_EXACT_NAMES = {hex_color: name for name, hex_color, _ in _COLORS}


def color_name(hex_color):
    hex_color = _normalize_hex(hex_color)
    if hex_color in _EXACT_NAMES:
        return _EXACT_NAMES[hex_color]
    target = _hex_to_oklab(hex_color)
    # min keeps the first entry on ties, preserving the supplied dataset order.
    return min(_COLORS, key=lambda entry: math.dist(target, entry[2]))[0]


class PixelGrid_HexToNames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hex_palette": ("STRING", {"multiline": True, "default": "#FF0000, #00FF00, #0000FF"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("color_names",)
    FUNCTION = "name_colors"
    CATEGORY = "Pixel Grid Helpers"

    def name_colors(self, hex_palette):
        hex_codes = [value.strip() for value in hex_palette.split(",") if value.strip()]
        names = dict.fromkeys(color_name(value) for value in hex_codes)
        return (", ".join(names),)
