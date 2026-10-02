import math
import re
from collections import Counter

import torch
import torch.nn.functional as F


_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_AUTO_COLOR_COUNTS = (8, 12, 16, 24, 32, 48, 64)
_SAMPLING_MODES = (
    "Exact pixels - existing pixel art",
    "Ink-preserving - thin dark lines",
    "Crisp shapes - clean color boundaries",
    "Area coverage - softer detail",
)
_COLOR_OPTIONS = (
    "2 colors - two-tone",
    "4 colors",
    "6 colors",
    "8 colors",
    "12 colors - recommended",
    "16 colors - detailed",
    "24 colors - rich detail",
    "32 colors - source-rich",
    "64 colors - extended",
    "128 colors",
    "256 colors - maximum",
)


def _parse_palette(value):
    if not value or not value.strip():
        return None

    colors = []
    seen = set()
    for item in value.split(","):
        color = item.strip()
        if not _HEX_COLOR.fullmatch(color):
            raise ValueError(
                f"Invalid palette color {color!r}. Use comma-separated #RRGGBB values."
            )
        normalized = color.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        colors.append(tuple(int(normalized[i : i + 2], 16) for i in (1, 3, 5)))

    if not colors:
        return None
    if len(colors) > 256:
        raise ValueError("Autofixer palettes may contain at most 256 colors.")
    return torch.tensor(colors, dtype=torch.uint8)


def _palette_string(palette):
    return ", ".join(
        "#{:02x}{:02x}{:02x}".format(*color)
        for color in palette.to(torch.uint8).tolist()
    )


def _luminance(colors):
    values = colors.to(torch.float64)
    return values[..., 0] * 0.299 + values[..., 1] * 0.587 + values[..., 2] * 0.114


def _detect_pixel_size(images):
    _, height, width, _ = images.shape
    if height < 2 or width < 2:
        return 1

    values = images.to(torch.int16)
    x_differences = (values[:, :, 1:] - values[:, :, :-1]).abs().amax(dim=-1)
    y_differences = (values[:, 1:] - values[:, :-1]).abs().amax(dim=-1)
    x_transitions = (x_differences > 48).sum(dim=(0, 1)).to(torch.int64)
    y_transitions = (y_differences > 48).sum(dim=(0, 2)).to(torch.int64)

    def periodic(axis, period):
        total = int(axis.sum())
        if not total:
            return False
        aligned = sum(
            int(axis[position - 1])
            for position in range(period, len(axis) + 1, period)
        )
        boundaries = sum(
            int(axis[position - 1] > 0)
            for position in range(period, len(axis) + 1, period)
        )
        return aligned / total > 0.985 and boundaries >= 4

    block = 1
    for candidate in range(2, min(24, height, width) + 1):
        if (
            width % candidate == 0
            and height % candidate == 0
            and periodic(x_transitions, candidate)
            and periodic(y_transitions, candidate)
        ):
            block = candidate
    return block


def _exact_downsample(images, pixel_size):
    if pixel_size == 1:
        return images.clone()

    _, height, width, _ = images.shape
    logical_height = math.ceil(height / pixel_size)
    logical_width = math.ceil(width / pixel_size)
    y_indices = (
        torch.arange(logical_height, dtype=torch.long) * pixel_size
        + pixel_size // 2
    ).clamp_max(height - 1)
    x_indices = (
        torch.arange(logical_width, dtype=torch.long) * pixel_size
        + pixel_size // 2
    ).clamp_max(width - 1)
    return images[:, y_indices][:, :, x_indices]


def _area_downsample(images, pixel_size):
    if pixel_size == 1:
        return images.clone()
    return (
        F.avg_pool2d(
            images.permute(0, 3, 1, 2).to(torch.float32),
            kernel_size=pixel_size,
            stride=pixel_size,
            ceil_mode=True,
            count_include_pad=False,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
    )


def _crisp_downsample(images, pixel_size, preserve_ink=False):
    if pixel_size == 1:
        return images.clone()

    batch, height, width, _ = images.shape
    logical_height = math.ceil(height / pixel_size)
    logical_width = math.ceil(width / pixel_size)
    pad_height = logical_height * pixel_size - height
    pad_width = logical_width * pixel_size - width
    padded = F.pad(
        images.permute(0, 3, 1, 2),
        (0, pad_width, 0, pad_height),
        mode="replicate",
    ).permute(0, 2, 3, 1)
    blocks = (
        padded.reshape(
            batch,
            logical_height,
            pixel_size,
            logical_width,
            pixel_size,
            3,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, logical_height, logical_width, pixel_size**2, 3)
    )

    keys = (
        (blocks[..., 0].to(torch.int32) >> 5) * 64
        + (blocks[..., 1].to(torch.int32) >> 5) * 8
        + (blocks[..., 2].to(torch.int32) >> 5)
    )
    dominant_key = torch.mode(keys, dim=-1).values
    dominant = keys == dominant_key.unsqueeze(-1)

    coordinates = torch.arange(pixel_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    center = (pixel_size - 1) / 2
    center_distance = ((xx - center).square() + (yy - center).square()).reshape(-1)
    family_distance = center_distance.expand_as(keys).masked_fill(~dominant, torch.inf)
    winner_index = family_distance.argmin(dim=-1)
    winner = torch.gather(
        blocks,
        3,
        winner_index[..., None, None].expand(*winner_index.shape, 1, 3),
    ).squeeze(3)

    if not preserve_ink:
        return winner

    block_luminance = _luminance(blocks)
    winner_luminance = _luminance(winner).unsqueeze(-1)
    ink = (block_luminance <= 80) & (winner_luminance - block_luminance >= 35)
    use_ink = ink.to(torch.float32).mean(dim=-1) >= 0.2
    ink_distance = center_distance.expand_as(keys).masked_fill(~ink, torch.inf)
    ink_index = ink_distance.argmin(dim=-1)
    ink_color = torch.gather(
        blocks,
        3,
        ink_index[..., None, None].expand(*ink_index.shape, 1, 3),
    ).squeeze(3)
    return torch.where(use_ink[..., None], ink_color, winner)


def _sample(images, pixel_size, sampling):
    if sampling == _SAMPLING_MODES[0]:
        return _exact_downsample(images, pixel_size)
    if sampling == _SAMPLING_MODES[1]:
        return _crisp_downsample(images, pixel_size, preserve_ink=True)
    if sampling == _SAMPLING_MODES[2]:
        return _crisp_downsample(images, pixel_size)
    if sampling == _SAMPLING_MODES[3]:
        return _area_downsample(images, pixel_size)
    raise ValueError(f"Unknown Autofixer sampling mode: {sampling!r}.")


def _palette_from_boxes(colors, counts, boxes):
    palette = []
    seen = set()
    for indices in boxes:
        weights = counts[indices].to(torch.float64)
        color = (
            (colors[indices].to(torch.float64) * weights[:, None]).sum(dim=0)
            / weights.sum()
        ).round().clamp(0, 255).to(torch.uint8)
        key = tuple(color.tolist())
        if key not in seen:
            seen.add(key)
            palette.append(color)
    return torch.stack(palette)


def _median_cut_palettes(colors, counts, limits):
    requested = set(limits)
    results = {}
    boxes = [torch.arange(len(colors), dtype=torch.long)]
    if 1 in requested:
        results[1] = _palette_from_boxes(colors, counts, boxes)
    while len(boxes) < max(limits):
        selected = None
        selected_axis = 0
        best_score = -1.0
        for box_index, indices in enumerate(boxes):
            if len(indices) < 2:
                continue
            box_colors = colors[indices]
            total = float(counts[indices].sum())
            for axis in range(3):
                spread = int(box_colors[:, axis].max()) - int(box_colors[:, axis].min())
                score = spread * math.sqrt(total)
                if score > best_score:
                    selected = box_index
                    selected_axis = axis
                    best_score = score

        if selected is None:
            break

        indices = boxes[selected]
        order = torch.argsort(colors[indices, selected_axis], stable=True)
        ordered = indices[order]
        cumulative = counts[ordered].cumsum(0)
        half = counts[ordered].sum().to(torch.float64) / 2
        cut = int(torch.searchsorted(cumulative.to(torch.float64), half).item()) + 1
        cut = min(max(1, cut), len(ordered) - 1)
        boxes[selected : selected + 1] = [ordered[:cut], ordered[cut:]]
        if len(boxes) in requested:
            results[len(boxes)] = _palette_from_boxes(colors, counts, boxes)

    fallback = _palette_from_boxes(colors, counts, boxes)
    for limit in limits:
        results.setdefault(limit, fallback)
    return results


def _palette_error(colors, counts, palette):
    distances = torch.cdist(
        colors.to(torch.float64), palette.to(torch.float64)
    ).square()
    nearest = distances.argmin(dim=1)
    mapped = palette[nearest].to(torch.float64)
    difference = colors.to(torch.float64) - mapped
    squared_error = (
        difference[:, 0].square() * 0.299
        + difference[:, 1].square() * 0.587
        + difference[:, 2].square() * 0.114
    )
    return math.sqrt(float((squared_error * counts).sum() / counts.sum()))


def _source_palette(images, max_colors=None):
    pixels = images.reshape(-1, 3)
    colors, counts = torch.unique(pixels, dim=0, return_counts=True, sorted=True)
    if max_colors is not None:
        if len(colors) <= max_colors:
            order = torch.argsort(_luminance(colors), stable=True)
            return colors[order]
        return _median_cut_palettes(colors, counts, (max_colors,))[max_colors]

    if len(colors) <= 256:
        order = torch.argsort(_luminance(colors), stable=True)
        return colors[order]

    horizontal = (
        images[:, :, 1:].to(torch.int16) - images[:, :, :-1].to(torch.int16)
    ).abs().amax(dim=-1)
    vertical = (
        images[:, 1:].to(torch.int16) - images[:, :-1].to(torch.int16)
    ).abs().amax(dim=-1)
    comparisons = horizontal.numel() + vertical.numel()
    flat = int((horizontal <= 3).sum()) + int((vertical <= 3).sum())
    tolerance = 7 if comparisons and flat / comparisons > 0.65 else 12

    palettes = _median_cut_palettes(colors, counts, _AUTO_COLOR_COUNTS)
    palette = palettes[_AUTO_COLOR_COUNTS[-1]]
    for count in _AUTO_COLOR_COUNTS:
        palette = palettes[count]
        if _palette_error(colors, counts, palette) <= tolerance:
            break
    return palette


def _map_palette(images, palette):
    flat = images.reshape(-1, 3).to(torch.float32)
    palette_float = palette.to(torch.float32)
    mapped = torch.empty_like(flat, dtype=torch.uint8)
    for start in range(0, len(flat), 65536):
        chunk = flat[start : start + 65536]
        nearest = torch.cdist(chunk, palette_float).argmin(dim=1)
        mapped[start : start + len(chunk)] = palette[nearest]
    return mapped.reshape(images.shape)


def _color_count(option):
    match = re.match(r"^(\d+) colors(?:\b| )", option)
    if not match:
        raise ValueError(f"Unknown Autofixer color option: {option!r}.")
    return int(match.group(1))


def _accent_capacity(color_count):
    if color_count < 4:
        return 0
    if color_count <= 6:
        return 1
    if color_count <= 8:
        return 2
    if color_count <= 12:
        return 3
    if color_count <= 24:
        return 4
    if color_count <= 64:
        return 6
    return 8


def _analyze_accent_frame(image):
    height, width, _ = image.shape
    pixel_count = height * width
    colors = [tuple(color) for color in image.reshape(-1, 3).tolist()]
    bins = [
        (color[0] >> 4) * 256 + (color[1] >> 4) * 16 + (color[2] >> 4)
        for color in colors
    ]
    seen = bytearray(pixel_count)
    grouped = {}
    lab_cache = {}
    ring = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    max_component = max(4, round(pixel_count * 0.025))

    def lab(color):
        if color not in lab_cache:
            lab_cache[color] = _oklab(color)
        return lab_cache[color]

    for seed in range(pixel_count):
        if seen[seed]:
            continue

        family = bins[seed]
        component = [seed]
        seen[seed] = 1
        head = 0
        touches_border = False
        while head < len(component):
            point = component[head]
            head += 1
            x = point % width
            y = point // width
            touches_border |= x == 0 or y == 0 or x == width - 1 or y == height - 1
            for dx, dy in ring:
                next_x = x + dx
                next_y = y + dy
                if not (0 <= next_x < width and 0 <= next_y < height):
                    continue
                neighbor = next_y * width + next_x
                if not seen[neighbor] and bins[neighbor] == family:
                    seen[neighbor] = 1
                    component.append(neighbor)

        if len(component) > max_component:
            continue

        members = set(component)
        surroundings = []
        for point in component:
            x = point % width
            y = point // width
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    next_x = x + dx
                    next_y = y + dy
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    neighbor = next_y * width + next_x
                    if neighbor not in members and bins[neighbor] != family:
                        surroundings.append(colors[neighbor])
        if len(surroundings) < 4:
            continue

        component_colors = [colors[point] for point in component]
        component_mean = tuple(
            sum(color[channel] for color in component_colors) / len(component_colors)
            for channel in range(3)
        )
        representative = min(
            component_colors,
            key=lambda color: (
                sum(
                    (color[channel] - component_mean[channel]) ** 2
                    for channel in range(3)
                ),
                color,
            ),
        )
        surrounding_mean = tuple(
            round(
                sum(color[channel] for color in surroundings) / len(surroundings)
            )
            for channel in range(3)
        )
        source_lab = lab(representative)
        surrounding_lab = lab(surrounding_mean)
        perceptual_contrast = _distance(source_lab, surrounding_lab)
        luminance_contrast = abs(source_lab[0] - surrounding_lab[0])
        source_chroma = math.hypot(source_lab[1], source_lab[2])
        surrounding_chroma = math.hypot(surrounding_lab[1], surrounding_lab[2])
        chroma_contrast = abs(source_chroma - surrounding_chroma)
        if (
            perceptual_contrast < 0.10
            and luminance_contrast < 0.09
            and chroma_contrast < 0.075
        ):
            continue

        coherence = min(1.0, math.sqrt(len(component)) / 3)
        rarity = 1 - min(1.0, len(component) / max(1, pixel_count * 0.025))
        score = (
            perceptual_contrast * (0.58 + 0.42 * coherence)
            + luminance_contrast * 0.24
            + chroma_contrast * 0.16
            + rarity * 0.035
            - (0.035 if touches_border else 0)
        )
        grouped.setdefault(family, []).append(
            {
                "points": component,
                "color": representative,
                "lab": source_lab,
                "score": score,
            }
        )

    candidates = []

    def add_candidate(components):
        strongest = max(
            components,
            key=lambda component: (
                component["score"],
                len(component["points"]),
                component["color"],
            ),
        )
        points = [
            point
            for component in components
            for point in component["points"]
        ]
        total = len(points)
        if len(components) == 1:
            xs = [point % width for point in points]
            ys = [point // width for point in points]
            span_x = max(xs) - min(xs) + 1
            span_y = max(ys) - min(ys) + 1
            if min(span_x, span_y) == 1 and max(span_x, span_y) >= 5:
                return
        candidates.append(
            {
                "points": points,
                "color": strongest["color"],
                "lab": strongest["lab"],
                "pixels": total,
                "score": strongest["score"]
                + min(0.045, 0.012 * (len(components) - 1)),
            }
        )

    for components in grouped.values():
        for component in components:
            if len(component["points"]) >= 2:
                add_candidate([component])

        singletons = [
            component for component in components if len(component["points"]) == 1
        ]
        if 2 <= len(singletons) <= 4:
            points = [component["points"][0] for component in singletons]
            xs = [point % width for point in points]
            ys = [point // width for point in points]
            if (
                max(xs) - min(xs) <= max(4, width * 0.35)
                and max(ys) - min(ys) <= max(4, height * 0.35)
            ):
                add_candidate(singletons)
    return candidates


def _accent_plan(images, color_count, reference_palette=None):
    capacity = _accent_capacity(color_count)
    batch, height, width, _ = images.shape
    assignments = torch.full((batch, height, width), -1, dtype=torch.int16)
    if not capacity:
        return torch.empty((0, 3), dtype=torch.uint8), assignments
    if reference_palette is None:
        reference_palette = _source_palette(images, color_count)
    reference_labs = [_oklab(tuple(color)) for color in reference_palette.tolist()]

    candidates = []
    for frame, image in enumerate(images):
        for candidate in _analyze_accent_frame(image):
            candidate["frame"] = frame
            candidate["representation_loss"] = min(
                _distance(candidate["lab"], palette_lab)
                for palette_lab in reference_labs
            )
            source_chroma = math.hypot(candidate["lab"][1], candidate["lab"][2])
            candidate["priority"] = (
                candidate["score"]
                + candidate["representation_loss"] * 1.2
                + min(0.25, source_chroma * 1.5)
                + min(0.05, 0.08 / math.sqrt(candidate["pixels"]))
            )
            candidates.append(candidate)
    candidates.sort(
        key=lambda candidate: (
            -candidate["priority"],
            candidate["color"],
            candidate["frame"],
        )
    )

    selected = []
    for candidate in candidates:
        if (
            candidate["score"] < 0.10
            or candidate["priority"] < 0.42
            or candidate["representation_loss"] < 0.055
        ):
            continue
        if any(_distance(candidate["lab"], accent["lab"]) < 0.065 for accent in selected):
            continue
        selected.append(candidate)
        if len(selected) == capacity:
            break
    if not selected:
        return torch.empty((0, 3), dtype=torch.uint8), assignments

    protected_counts = [[0] * len(selected) for _ in range(batch)]
    protected_limit = max(6, round(height * width * 0.025))
    selected_indices = {id(accent): index for index, accent in enumerate(selected)}
    for candidate in candidates:
        if (
            candidate["score"] < 0.10
            or candidate["priority"] < 0.42
            or candidate["representation_loss"] < 0.055
        ):
            continue
        target = selected_indices.get(id(candidate))
        if target is None:
            distances = [
                _distance(candidate["lab"], accent["lab"])
                for accent in selected
            ]
            target = min(range(len(distances)), key=distances.__getitem__)
            if (
                distances[target] > 0.035
                or distances[target] >= candidate["representation_loss"]
            ):
                continue

        frame = candidate["frame"]
        remaining = protected_limit - protected_counts[frame][target]
        if remaining <= 0:
            continue
        flat_assignments = assignments[frame].reshape(-1)
        points = torch.tensor(candidate["points"], dtype=torch.long)
        points = points[flat_assignments[points] < 0][:remaining]
        flat_assignments[points] = target
        protected_counts[frame][target] += len(points)

    used = [
        index
        for index in range(len(selected))
        if bool((assignments == index).any())
    ]
    if len(used) != len(selected):
        remapped = torch.full_like(assignments, -1)
        for next_index, previous_index in enumerate(used):
            remapped[assignments == previous_index] = next_index
        assignments = remapped
        selected = [selected[index] for index in used]
    colors = torch.tensor([accent["color"] for accent in selected], dtype=torch.uint8)
    return colors, assignments


def _linear_srgb(value):
    normalized = value / 255.0
    if normalized <= 0.04045:
        return normalized / 12.92
    return ((normalized + 0.055) / 1.055) ** 2.4


def _oklab(color):
    red, green, blue = (_linear_srgb(value) for value in color)
    light = math.cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue)
    medium = math.cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue)
    short = math.cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue)
    return (
        0.2104542553 * light + 0.793617785 * medium - 0.0040720468 * short,
        1.9779984951 * light - 2.428592205 * medium + 0.4505937099 * short,
        0.0259040371 * light + 0.7827717662 * medium - 0.808675766 * short,
    )


def _distance(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _clean_small_islands(image, protection=None):
    height, width, _ = image.shape
    colors = [tuple(color) for color in image.reshape(-1, 3).tolist()]
    output = colors.copy()
    protected = (
        protection.reshape(-1).tolist()
        if protection is not None
        else [False] * (width * height)
    )
    seen = bytearray(width * height)
    labs = {}
    ring = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    def lab(color):
        if color not in labs:
            labs[color] = _oklab(color)
        return labs[color]

    for seed in range(width * height):
        if seen[seed]:
            continue

        component_color = colors[seed]
        component = [seed]
        seen[seed] = 1
        head = 0
        min_x = max_x = seed % width
        min_y = max_y = seed // width
        while head < len(component):
            point = component[head]
            head += 1
            x = point % width
            y = point // width
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for dx, dy in ring:
                next_x = x + dx
                next_y = y + dy
                if not (0 <= next_x < width and 0 <= next_y < height):
                    continue
                neighbor = next_y * width + next_x
                if not seen[neighbor] and colors[neighbor] == component_color:
                    seen[neighbor] = 1
                    component.append(neighbor)

        if len(component) > 3:
            continue
        if any(protected[point] for point in component):
            continue
        if min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1:
            continue
        if len(component) >= 3 and max(max_x - min_x, max_y - min_y) >= len(component) - 1:
            continue

        members = set(component)
        boundary = Counter()
        patch = set()
        contacts = 0
        for point in component:
            x = point % width
            y = point // width
            for dx, dy in ring:
                neighbor = (y + dy) * width + x + dx
                if neighbor in members or protected[neighbor]:
                    continue
                weight = 1 if dx and dy else 2
                contacts += weight
                boundary[colors[neighbor]] += weight
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    next_x = x + dx
                    next_y = y + dy
                    if 0 <= next_x < width and 0 <= next_y < height:
                        neighbor = next_y * width + next_x
                        if neighbor not in members and not protected[neighbor]:
                            patch.add(neighbor)

        if not contacts or not patch:
            continue
        ranked = sorted(boundary.items(), key=lambda item: (-item[1], item[0]))
        target, votes = ranked[0]
        support = votes / contacts
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if support < 0.72 or (votes - runner_up) / contacts < 0.18:
            continue

        source_lab = lab(component_color)
        target_lab = lab(target)
        perceptual_distance = _distance(source_lab, target_lab)
        if perceptual_distance > 0.065:
            continue
        chroma_distance = math.hypot(
            source_lab[1] - target_lab[1], source_lab[2] - target_lab[2]
        )
        if (
            (source_lab[0] < 0.32 or source_lab[0] > 0.92)
            and perceptual_distance > 0.055
        ) or (perceptual_distance > 0.10 and chroma_distance > 0.045):
            continue

        local_support = sum(colors[point] == target for point in patch) / len(patch)
        confidence = (
            0.55 * support
            + 0.25 * local_support
            + 0.20 * max(0.0, 1 - perceptual_distance / (0.065 * 1.35))
        )
        if local_support < 0.58 or confidence < 0.70:
            continue
        for point in component:
            output[point] = target

    return torch.tensor(output, dtype=torch.uint8).reshape(height, width, 3)


class PixelGrid_Autofixer:
    """Deterministic grid recovery, accent-aware palettes, and conservative cleanup."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "pixel_size": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 128,
                        "tooltip": "0 detects enlarged pixel blocks automatically; 1 keeps the native grid.",
                    },
                ),
                "palette": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional comma-separated #RRGGBB palette.",
                    },
                ),
                "sampling": (
                    _SAMPLING_MODES,
                    {
                        "default": _SAMPLING_MODES[1],
                        "tooltip": "How each source pixel block becomes one logical pixel.",
                    },
                ),
                "colors": (
                    _COLOR_OPTIONS,
                    {
                        "default": "12 colors - recommended",
                        "tooltip": "Maximum generated palette size. A custom palette overrides this setting.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("fixed_image", "hex_palette", "pixel_size_used")
    FUNCTION = "autofix"
    CATEGORY = "Pixel Grid Helpers"
    DESCRIPTION = (
        "Recovers the native pixel grid, preserves coherent small accents such as "
        "eyes and jewelry within the color budget, and removes only high-confidence "
        "tiny color islands."
    )

    def autofix(
        self,
        image,
        pixel_size=0,
        palette="",
        sampling=_SAMPLING_MODES[1],
        colors="12 colors - recommended",
    ):
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Autofixer expects a ComfyUI IMAGE tensor shaped [B, H, W, 3].")
        if not image.shape[0] or not image.shape[1] or not image.shape[2]:
            raise ValueError("Autofixer received an empty image.")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("Autofixer does not accept NaN or infinite image values.")
        if not isinstance(pixel_size, int) or pixel_size < 0 or pixel_size > 128:
            raise ValueError("pixel_size must be an integer from 0 to 128.")

        color_count = _color_count(colors)
        device = image.device
        output_dtype = image.dtype
        source = (
            image.detach()
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
        )
        resolved_size = pixel_size or _detect_pixel_size(source)
        logical = _sample(source, resolved_size, sampling)
        selected_palette = _parse_palette(palette)
        custom_palette = selected_palette is not None
        accent_limit = len(selected_palette) if selected_palette is not None else color_count
        provisional_palette = (
            selected_palette
            if selected_palette is not None
            else _source_palette(logical, color_count)
        )
        accent_colors, accent_assignments = _accent_plan(
            logical, accent_limit, provisional_palette
        )
        protection = accent_assignments >= 0

        if selected_palette is None:
            if len(accent_colors):
                base_slots = max(1, color_count - len(accent_colors))
                base_pixels = logical[~protection]
                if not len(base_pixels):
                    base_pixels = logical.reshape(-1, 3)
                base_palette = _source_palette(base_pixels, base_slots)
                accent_keys = {tuple(color) for color in accent_colors.tolist()}
                keep = [
                    tuple(color) not in accent_keys
                    for color in base_palette.tolist()
                ]
                base_palette = base_palette[torch.tensor(keep, dtype=torch.bool)]
                quantized = (
                    _map_palette(logical, base_palette)
                    if len(base_palette)
                    else _map_palette(logical, accent_colors)
                )
                for index, color in enumerate(accent_colors):
                    quantized[accent_assignments == index] = color
                selected_palette = (
                    torch.cat((base_palette, accent_colors))
                    if len(base_palette)
                    else accent_colors
                )
            else:
                selected_palette = provisional_palette
                quantized = _map_palette(logical, selected_palette)
        else:
            quantized = _map_palette(logical, selected_palette)
            if len(accent_colors):
                mapped_accents = _map_palette(
                    accent_colors.reshape(1, 1, -1, 3), selected_palette
                ).reshape(-1, 3)
                for index, color in enumerate(mapped_accents):
                    quantized[accent_assignments == index] = color

        cleaned = torch.stack(
            [
                _clean_small_islands(frame, protection[index])
                for index, frame in enumerate(quantized)
            ]
        )
        if not custom_palette:
            used_colors = {
                tuple(color)
                for color in torch.unique(cleaned.reshape(-1, 3), dim=0).tolist()
            }
            keep = [
                tuple(color) in used_colors
                for color in selected_palette.tolist()
            ]
            selected_palette = selected_palette[torch.tensor(keep, dtype=torch.bool)]

        if resolved_size > 1:
            restored = (
                cleaned.repeat_interleave(resolved_size, dim=1)
                .repeat_interleave(resolved_size, dim=2)[
                    :, : source.shape[1], : source.shape[2]
                ]
                .to(torch.float32)
            )
        else:
            restored = cleaned.to(torch.float32)

        result = restored.div(255).to(device=device, dtype=output_dtype)
        return result, _palette_string(selected_palette), resolved_size
