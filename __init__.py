from .nodes import (
    PixelGrid_KMeans,
    PixelGrid_ApplyPalette,
    PixelGrid_MergeSimilar,
    PixelGrid_Analyze,
    PixelGrid_PaletteToImage,
    GridMedianFixer
)
from .autofixer import PixelGrid_Autofixer
from .color_names import PixelGrid_HexToNames

NODE_CLASS_MAPPINGS = {
    "PixelGrid_KMeans": PixelGrid_KMeans,
    "PixelGrid_ApplyPalette": PixelGrid_ApplyPalette,
    "PixelGrid_MergeSimilar": PixelGrid_MergeSimilar,
    "PixelGrid_Analyze": PixelGrid_Analyze,
    "PixelGrid_PaletteToImage": PixelGrid_PaletteToImage,
    "GridMedianFixer": GridMedianFixer,
    "PixelGrid_Autofixer": PixelGrid_Autofixer,
    "PixelGrid_HexToNames": PixelGrid_HexToNames
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelGrid_KMeans": "Pixel Grid: Quantize (Max Colors)",
    "PixelGrid_ApplyPalette": "Pixel Grid: Enforce Palette",
    "PixelGrid_MergeSimilar": "Pixel Grid: Merge Similar Colors",
    "PixelGrid_Analyze": "Pixel Grid: Analyze Palette",
    "PixelGrid_PaletteToImage": "Pixel Grid: Palette to Image",
    "GridMedianFixer": "Pixel Grid: Median Fixer",
    "PixelGrid_Autofixer": "Pixel Grid: Autofixer",
    "PixelGrid_HexToNames": "Pixel Grid: HEX to Color Names"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]