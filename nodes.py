import torch
import re

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

class PixelGrid_CreateSolidColorImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "hex_code": ("STRING", {"default": "#FF0000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_image"
    CATEGORY = "Pixel Grid Helpers"

    def create_image(self, width, height, hex_code):
        match = re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", hex_code.strip())
        if match is None:
            raise ValueError(
                f"Invalid HEX color {hex_code!r}; use 3 or 6 hexadecimal digits."
            )

        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(digit * 2 for digit in digits)
        rgb = [int(digits[index:index + 2], 16) for index in (0, 2, 4)]
        color = torch.tensor(rgb, dtype=torch.float32).div_(255).view(1, 1, 1, 3)
        image = color.expand(1, height, width, 3).clone()
        return (image,)

class PixelGrid_KMeans:
    """Node 1: Enforce Max N Colors (K-Means Quantization)"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_colors": ("INT", {"default": 16, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("quantized_image", "hex_palette")
    FUNCTION = "quantize"
    CATEGORY = "Pixel Grid Helpers"

    def quantize(self, image, max_colors):
        # image: [B, H, W, 3]
        # Flatten: [N, 3]
        pixels = image.reshape(-1, 3)
        
        # Simple K-Means implementation in Torch
        # Initialize centroids randomly from existing pixels
        num_pixels = pixels.shape[0]
        indices = torch.randperm(num_pixels)[:max_colors]
        centroids = pixels[indices]

        for _ in range(10): # 10 iterations is usually enough for pixel art
            # Calculate distances [Pixels, Centroids]
            dists = torch.cdist(pixels, centroids)
            # Assign to nearest
            labels = torch.argmin(dists, dim=1)
            # Update centroids
            new_centroids = []
            for i in range(max_colors):
                mask = (labels == i)
                if mask.any():
                    new_centroids.append(pixels[mask].mean(dim=0))
                else:
                    # If a centroid has no pixels, re-init it random
                    random_idx = torch.randint(0, num_pixels, (1,))
                    new_centroids.append(pixels[random_idx].squeeze())
            centroids = torch.stack(new_centroids)

        # Final assignment
        dists = torch.cdist(pixels, centroids)
        labels = torch.argmin(dists, dim=1)
        
        # Reconstruct
        quantized = centroids[labels].reshape(image.shape)
        
        # Generate Palette String
        # Convert 0-1 float rgb to 0-255 int
        palette_rgb = (centroids * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        hex_list = [rgb_to_hex(c) for c in palette_rgb]
        hex_string = ", ".join(hex_list)

        return (quantized, hex_string)

class PixelGrid_ApplyPalette:
    """Node 2: Enforce Provided Color Palette"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "hex_palette": ("STRING", {"multiline": True, "default": "#000000, #FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("quantized_image", "hex_palette_out")
    FUNCTION = "apply_palette"
    CATEGORY = "Pixel Grid Helpers"

    def apply_palette(self, image, hex_palette):
        # Parse hex string
        hex_codes = [x.strip() for x in hex_palette.split(',') if x.strip()]
        if not hex_codes:
            return (image, hex_palette)

        # Create Palette Tensor [K, 3]
        rgb_list = [hex_to_rgb(h) for h in hex_codes]
        palette = torch.tensor(rgb_list, dtype=torch.float32, device=image.device) / 255.0

        # Flatten image
        pixels = image.reshape(-1, 3)
        
        # Find nearest color
        dists = torch.cdist(pixels, palette) # [N_pixels, N_palette]
        labels = torch.argmin(dists, dim=1)
        
        # Map
        new_pixels = palette[labels]
        result = new_pixels.reshape(image.shape)

        return (result, hex_palette)

class PixelGrid_MergeSimilar:
    """Node 3: Merge Similar Colors by Threshold"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("merged_image", "hex_palette")
    FUNCTION = "merge_colors"
    CATEGORY = "Pixel Grid Helpers"

    def merge_colors(self, image, threshold):
        # 1. Get Unique Colors
        pixels = image.reshape(-1, 3)
        unique_colors, inverse_indices = torch.unique(pixels, dim=0, return_inverse=True)
        
        # 2. Greedy Merging
        # We will iterate and merge colors that are close
        final_palette = [] # List of tensors
        
        # We use a mask to track processed colors
        processed = torch.zeros(unique_colors.shape[0], dtype=torch.bool, device=image.device)
        
        for i in range(unique_colors.shape[0]):
            if processed[i]:
                continue
            
            current_color = unique_colors[i]
            
            # Calculate distance to all unprocessed colors
            dists = torch.norm(unique_colors - current_color, dim=1)
            
            # Find close colors (including itself)
            mask = (dists <= threshold) & (~processed)
            
            # Average these colors
            group = unique_colors[mask]
            merged_color = group.mean(dim=0)
            
            final_palette.append(merged_color)
            
            # Mark as processed
            processed = processed | mask
            
            # Update the unique_colors array for the remapping step
            # (We overwrite the original unique colors with the new merged one so mapping works)
            unique_colors[mask] = merged_color

        # 3. Reconstruct
        # inverse_indices points to the original unique_colors table, which we just updated in place
        new_pixels = unique_colors[inverse_indices]
        result = new_pixels.reshape(image.shape)
        
        # 4. Generate Palette String
        palette_tensor = torch.stack(final_palette)
        palette_rgb = (palette_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        hex_list = [rgb_to_hex(c) for c in palette_rgb]
        
        return (result, ", ".join(hex_list))

class PixelGrid_Analyze:
    """Node 4: Analyze Color Palette"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "image": ("IMAGE",), }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hex_palette",)
    FUNCTION = "analyze"
    CATEGORY = "Pixel Grid Helpers"

    def analyze(self, image):
        pixels = image.reshape(-1, 3)
        unique_colors = torch.unique(pixels, dim=0)
        
        palette_rgb = (unique_colors * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        hex_list = [rgb_to_hex(c) for c in palette_rgb]
        return (", ".join(hex_list),)

class PixelGrid_PaletteToImage:
    """Node 5: Palette to Image Swatches"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "hex_palette": ("STRING", {"multiline": True, "default": "#FF0000, #00FF00"}), }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("swatch_image",)
    FUNCTION = "generate"
    CATEGORY = "Pixel Grid Helpers"

    def generate(self, hex_palette):
        hex_codes = [x.strip() for x in hex_palette.split(',') if x.strip()]
        if not hex_codes:
             # Return black 24x24 if empty
             return (torch.zeros((1, 24, 24, 3)),)

        swatches = []
        for h in hex_codes:
            rgb = hex_to_rgb(h)
            # Create [24, 24, 3] tensor
            # Normalize to 0-1
            tensor_color = torch.tensor(rgb, dtype=torch.float32) / 255.0
            patch = tensor_color.view(1, 1, 3).repeat(24, 24, 1)
            swatches.append(patch)
        
        # Stack vertically [H*Count, W, 3]
        full_image = torch.cat(swatches, dim=0)
        # Add batch dim [1, H, W, 3]
        full_image = full_image.unsqueeze(0)
        
        return (full_image,)

# Re-including the previous node requested
class GridMedianFixer:
    """Median-reduce an image using the most coherent pixel-grid phase."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_size": ("INT", {"default": 6, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("downscaled_image", "original_size_image")
    FUNCTION = "process_grid"
    CATEGORY = "Pixel Grid Helpers"

    @staticmethod
    def _phase_downscaled(image, grid_size, y_offset, x_offset):
        batch, height, width, channels = image.shape
        rows = max((height - y_offset) // grid_size, 0)
        columns = max((width - x_offset) // grid_size, 0)
        end_y = y_offset + rows * grid_size
        end_x = x_offset + columns * grid_size
        cropped = image[:, y_offset:end_y, x_offset:end_x, :]

        blocks = cropped.reshape(
            batch, rows, grid_size, columns, grid_size, channels
        ).permute(0, 1, 3, 2, 4, 5)
        blocks = blocks.reshape(
            batch, rows, columns, grid_size * grid_size, channels
        )
        return torch.median(blocks, dim=3).values

    @classmethod
    def _phase_outputs(cls, image, grid_size, y_offset, x_offset):
        downscaled = cls._phase_downscaled(
            image, grid_size, y_offset, x_offset
        )
        upscaled = downscaled.repeat_interleave(grid_size, dim=1).repeat_interleave(
            grid_size, dim=2
        )
        return downscaled, upscaled

    @staticmethod
    def _common_phase_bounds(length, grid_size):
        valid_offsets = [
            offset
            for offset in range(grid_size)
            if length - offset >= grid_size
        ]
        if not valid_offsets:
            return 0, 0

        start = valid_offsets[-1]
        end = min(
            offset + ((length - offset) // grid_size) * grid_size
            for offset in valid_offsets
        )
        return start, end

    @staticmethod
    def _phase_error(
        image,
        grid_size,
        y_offset,
        x_offset,
        score_top,
        score_bottom,
        score_left,
        score_right,
    ):
        batch, _, _, channels = image.shape
        first_row = (score_top - y_offset) // grid_size
        last_row = (score_bottom - 1 - y_offset) // grid_size + 1
        first_column = (score_left - x_offset) // grid_size
        last_column = (score_right - 1 - x_offset) // grid_size + 1

        block_top = y_offset + first_row * grid_size
        block_bottom = y_offset + last_row * grid_size
        block_left = x_offset + first_column * grid_size
        block_right = x_offset + last_column * grid_size
        rows = last_row - first_row
        columns = last_column - first_column
        score_pixels = image[:, block_top:block_bottom, block_left:block_right, :]
        blocks = score_pixels.reshape(
            batch, rows, grid_size, columns, grid_size, channels
        ).permute(0, 1, 3, 2, 4, 5)
        blocks = blocks.reshape(
            batch, rows, columns, grid_size * grid_size, channels
        )
        score_medians = torch.median(blocks, dim=3).values
        aligned_region = score_medians.repeat_interleave(
            grid_size, dim=1
        ).repeat_interleave(grid_size, dim=2)

        local_top = score_top - block_top
        local_left = score_left - block_left
        region_height = score_bottom - score_top
        region_width = score_right - score_left
        aligned_region = aligned_region[
            :, local_top:local_top + region_height,
            local_left:local_left + region_width, :
        ]
        source_region = image[:, score_top:score_bottom, score_left:score_right, :]
        error = source_region.to(torch.float32) - aligned_region.to(torch.float32)
        return error.abs().mean()

    def process_grid(self, image, grid_size):
        if grid_size < 1:
            raise ValueError("grid_size must be at least 1")

        _, height, width, _ = image.shape
        with torch.no_grad():
            if height < grid_size or width < grid_size:
                return self._phase_outputs(image, grid_size, 0, 0)

            score_top, score_bottom = self._common_phase_bounds(height, grid_size)
            score_left, score_right = self._common_phase_bounds(width, grid_size)
            phases = []
            scores = []

            y_offsets = range(min(grid_size, height - grid_size + 1))
            x_offsets = range(min(grid_size, width - grid_size + 1))
            for y_offset in y_offsets:
                for x_offset in x_offsets:
                    score = self._phase_error(
                        image,
                        grid_size,
                        y_offset,
                        x_offset,
                        score_top,
                        score_bottom,
                        score_left,
                        score_right,
                    )
                    phases.append((y_offset, x_offset))
                    scores.append(score)

            best_index = torch.argmin(torch.stack(scores)).item()
            best_y, best_x = phases[best_index]
            return self._phase_outputs(image, grid_size, best_y, best_x)