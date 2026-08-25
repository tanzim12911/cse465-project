"""Deterministic image analysis tools for Color Illusion reasoning."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class ColorAnalysisTool:
    """Compute lightweight, deterministic color statistics from an RGB image.

    Output is designed to be injected into Generator/Solver prompts as
    auxiliary evidence.  No model weights are modified.
    """

    def __init__(self, num_dominant_colors: int = 3, luminance_bins: int = 4):
        self.num_dominant_colors = num_dominant_colors
        self.luminance_bins = luminance_bins

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3)

        mean_rgb = self._mean_rgb(pixels)
        dominant = self._dominant_colors(pixels)
        lum_hist = self._luminance_histogram(pixels, self.luminance_bins)
        var_rgb = self._variance(pixels)

        return {
            "mean_rgb": mean_rgb,
            "dominant_colors": dominant,
            "luminance_histogram": lum_hist,
            "rgb_variance": var_rgb,
            "pixel_count": int(pixels.shape[0]),
        }

    def format_for_prompt(self, image: Image.Image, label: str = "image") -> str:
        stats = self.analyze(image)
        lines = [
            f"[Color Analysis Tool — {label}]",
            f"Mean RGB: R={stats['mean_rgb'][0]:.1f} G={stats['mean_rgb'][1]:.1f} B={stats['mean_rgb'][2]:.1f}",
            f"RGB variance: R={stats['rgb_variance'][0]:.1f} G={stats['rgb_variance'][1]:.1f} B={stats['rgb_variance'][2]:.1f}",
            f"Dominant colors (RGB):",
        ]
        for i, (r, g, b) in enumerate(stats["dominant_colors"], 1):
            lines.append(f"  {i}. ({r:.1f}, {g:.1f}, {b:.1f})")
        lines.append("Luminance distribution (low → high):")
        for i, (low, high, count, pct) in enumerate(stats["luminance_histogram"], 1):
            lines.append(f"  Bin {i} [{low:.2f}-{high:.2f}]: {pct:.0f}% of pixels")
        return "\n".join(lines)

    @staticmethod
    def _mean_rgb(pixels: np.ndarray) -> Tuple[float, float, float]:
        m = pixels.mean(axis=0)
        return (float(m[0]), float(m[1]), float(m[2]))

    @staticmethod
    def _variance(pixels: np.ndarray) -> Tuple[float, float, float]:
        v = pixels.var(axis=0)
        return (float(v[0]), float(v[1]), float(v[2]))

    def _dominant_colors(self, pixels: np.ndarray) -> List[Tuple[float, float, float]]:
        # Simple quantization to 32 levels per channel for fast clustering.
        quantized = (pixels / 32).astype(np.int32)
        keys = quantized[:, 0] * 4096 + quantized[:, 1] * 64 + quantized[:, 2]
        keys_list = keys.tolist()
        counts: Dict[int, int] = {}
        for k in keys_list:
            counts[k] = counts.get(k, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: self.num_dominant_colors]
        result = []
        for k, _ in top:
            r = (k // 4096) * 32 + 16
            g = ((k % 4096) // 64) * 32 + 16
            b = (k % 64) * 32 + 16
            result.append((float(r), float(g), float(b)))
        return result

    def _luminance_histogram(
        self, pixels: np.ndarray, bins: int
    ) -> List[Tuple[float, float, int, float]]:
        lum = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        lum = lum / 255.0
        counts, edges = np.histogram(lum, bins=bins, range=(0.0, 1.0))
        total = float(counts.sum()) if counts.sum() > 0 else 1.0
        result = []
        for i in range(bins):
            low = float(edges[i])
            high = float(edges[i + 1])
            count = int(counts[i])
            pct = count / total * 100.0
            result.append((low, high, count, pct))
        return result
