"""
damage_analyzer.py

Converts binary segmentation masks from the CV models into structured
text descriptions that the LLM can use to generate accurate claim reports.
"""

import numpy as np
import cv2
from typing import List, Dict


def _get_spatial_location(cx: float, cy: float, img_w: int, img_h: int) -> str:
    """
    Map a centroid (cx, cy) to a human-readable spatial location
    using a 3x3 grid over the image.
    """
    # Vertical position
    if cy < img_h * 0.33:
        v = "upper"
    elif cy < img_h * 0.66:
        v = "center"
    else:
        v = "lower"

    # Horizontal position
    if cx < img_w * 0.33:
        h = "left"
    elif cx < img_w * 0.66:
        h = "center"
    else:
        h = "right"

    if v == "center" and h == "center":
        return "center"
    elif v == "center":
        return h
    elif h == "center":
        return v
    else:
        return f"{v}-{h}"


def _estimate_severity(damage_pct: float) -> str:
    """
    Estimate damage severity based on the percentage of the image
    covered by the damage mask.
    """
    if damage_pct < 2.0:
        return "Minor"
    elif damage_pct < 10.0:
        return "Moderate"
    else:
        return "Severe"


def analyze_single_mask(mask: np.ndarray, image_index: int = 1) -> Dict:
    """
    Analyze a single binary segmentation mask and return structured info.

    Args:
        mask: Binary numpy array (0 = no damage, 1 = damage).
        image_index: 1-based index for labeling.

    Returns:
        Dict with damage analysis results.
    """
    h, w = mask.shape[:2]
    total_pixels = h * w
    damage_pixels = int(np.sum(mask > 0))
    damage_pct = (damage_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    result = {
        "image_index": image_index,
        "image_resolution": f"{w}x{h}",
        "damage_detected": damage_pixels > 0,
        "damage_area_pct": round(damage_pct, 2),
        "severity": _estimate_severity(damage_pct),
        "regions": [],
    }

    if damage_pixels == 0:
        return result

    # Find connected components (separate damage regions)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    regions = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Skip tiny noise regions (less than 0.1% of image)
        if area < total_pixels * 0.001:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        rw = int(stats[i, cv2.CC_STAT_WIDTH])
        rh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        region_pct = (area / total_pixels) * 100
        location = _get_spatial_location(cx, cy, w, h)

        regions.append({
            "region_id": len(regions) + 1,
            "location": location,
            "bounding_box": {"x": x, "y": y, "width": rw, "height": rh},
            "area_pct": round(region_pct, 2),
            "severity": _estimate_severity(region_pct),
        })

    # Sort by area (largest damage region first)
    regions.sort(key=lambda r: r["area_pct"], reverse=True)
    result["regions"] = regions
    result["num_damage_regions"] = len(regions)

    return result


def analyze_masks(masks: List[np.ndarray]) -> List[Dict]:
    """
    Analyze multiple masks and return a list of analysis results.
    """
    return [analyze_single_mask(m, i + 1) for i, m in enumerate(masks)]


def format_damage_report(analyses: List[Dict]) -> str:
    """
    Convert mask analysis results into a structured text description
    suitable for the LLM's detected_damage prompt variable.

    Args:
        analyses: List of dicts from analyze_masks().

    Returns:
        Formatted string describing all detected damage.
    """
    if not analyses:
        return "No images were provided for damage analysis."

    lines = []
    lines.append(f"DAMAGE ANALYSIS SUMMARY ({len(analyses)} image(s) analyzed)")
    lines.append("=" * 60)

    total_with_damage = sum(1 for a in analyses if a["damage_detected"])
    lines.append(f"Images with detected damage: {total_with_damage}/{len(analyses)}")
    lines.append("")

    for a in analyses:
        lines.append(f"--- Image {a['image_index']} ({a['image_resolution']}) ---")

        if not a["damage_detected"]:
            lines.append("  No damage detected in this image.")
            lines.append("")
            continue

        lines.append(f"  Overall damage coverage: {a['damage_area_pct']}% of image area")
        lines.append(f"  Overall severity: {a['severity']}")
        lines.append(f"  Number of distinct damage regions: {a.get('num_damage_regions', 0)}")

        for r in a["regions"]:
            lines.append(
                f"    Region {r['region_id']}: "
                f"Location = {r['location']}, "
                f"Area = {r['area_pct']}%, "
                f"Severity = {r['severity']}, "
                f"Bounding box = ({r['bounding_box']['x']}, {r['bounding_box']['y']}, "
                f"{r['bounding_box']['width']}x{r['bounding_box']['height']})"
            )

        lines.append("")

    return "\n".join(lines)


def generate_detected_damage(masks: List[np.ndarray]) -> str:
    """
    One-call convenience function: takes masks from the segmentation models
    and returns a formatted damage description string ready for the LLM.

    Args:
        masks: List of binary numpy arrays from segment_damage().

    Returns:
        Formatted damage description string.
    """
    analyses = analyze_masks(masks)
    return format_damage_report(analyses)