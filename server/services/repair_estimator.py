"""
repair_estimator.py

Estimates repair costs based on damage analysis output from the CV pipeline.
Uses a lookup table of average repair costs by vehicle panel and severity,
sourced from industry averages (CCC, Mitchell, Audatex benchmarks).

This is a simplified estimator for demonstration purposes. Production systems
would integrate with live parts catalogs and regional labor rate databases.
"""

from typing import Dict, List, Optional, Tuple
import re


# Average repair costs by panel and severity (USD).
# Sources: Industry averages from auto body repair guides, 2024-2025.
# Each entry: (labor_low, labor_high, parts_low, parts_high)
_REPAIR_COSTS = {
    "front_bumper": {
        "Minor":    {"labor": (150, 300),   "parts": (50, 200),    "paint": (100, 200)},
        "Moderate": {"labor": (300, 600),   "parts": (200, 500),   "paint": (200, 400)},
        "Severe":   {"labor": (500, 1000),  "parts": (400, 1200),  "paint": (300, 500)},
    },
    "rear_bumper": {
        "Minor":    {"labor": (150, 300),   "parts": (50, 200),    "paint": (100, 200)},
        "Moderate": {"labor": (300, 600),   "parts": (200, 500),   "paint": (200, 400)},
        "Severe":   {"labor": (500, 1000),  "parts": (400, 1200),  "paint": (300, 500)},
    },
    "hood": {
        "Minor":    {"labor": (200, 400),   "parts": (100, 300),   "paint": (150, 300)},
        "Moderate": {"labor": (400, 800),   "parts": (300, 800),   "paint": (300, 500)},
        "Severe":   {"labor": (600, 1200),  "parts": (500, 1500),  "paint": (400, 600)},
    },
    "trunk": {
        "Minor":    {"labor": (150, 350),   "parts": (100, 250),   "paint": (100, 250)},
        "Moderate": {"labor": (350, 700),   "parts": (250, 600),   "paint": (250, 400)},
        "Severe":   {"labor": (500, 1000),  "parts": (400, 1200),  "paint": (300, 500)},
    },
    "front_fender": {
        "Minor":    {"labor": (200, 400),   "parts": (100, 300),   "paint": (150, 300)},
        "Moderate": {"labor": (400, 700),   "parts": (250, 600),   "paint": (250, 400)},
        "Severe":   {"labor": (600, 1100),  "parts": (400, 1000),  "paint": (350, 550)},
    },
    "rear_fender": {
        "Minor":    {"labor": (250, 500),   "parts": (100, 300),   "paint": (150, 300)},
        "Moderate": {"labor": (500, 900),   "parts": (300, 700),   "paint": (300, 500)},
        "Severe":   {"labor": (800, 1500),  "parts": (500, 1200),  "paint": (400, 600)},
    },
    "quarter_panel": {
        "Minor":    {"labor": (300, 600),   "parts": (150, 400),   "paint": (200, 350)},
        "Moderate": {"labor": (600, 1200),  "parts": (400, 900),   "paint": (350, 550)},
        "Severe":   {"labor": (1000, 2000), "parts": (600, 1800),  "paint": (500, 700)},
    },
    "door": {
        "Minor":    {"labor": (200, 400),   "parts": (100, 300),   "paint": (150, 300)},
        "Moderate": {"labor": (400, 800),   "parts": (300, 700),   "paint": (300, 450)},
        "Severe":   {"labor": (600, 1200),  "parts": (500, 1500),  "paint": (400, 600)},
    },
    "roof": {
        "Minor":    {"labor": (300, 600),   "parts": (100, 300),   "paint": (200, 400)},
        "Moderate": {"labor": (600, 1200),  "parts": (300, 800),   "paint": (400, 600)},
        "Severe":   {"labor": (1000, 2500), "parts": (500, 2000),  "paint": (500, 800)},
    },
    "windshield": {
        "Minor":    {"labor": (50, 100),    "parts": (100, 250),   "paint": (0, 0)},
        "Moderate": {"labor": (100, 200),   "parts": (200, 500),   "paint": (0, 0)},
        "Severe":   {"labor": (150, 300),   "parts": (250, 800),   "paint": (0, 0)},
    },
    "side_mirror": {
        "Minor":    {"labor": (50, 100),    "parts": (50, 150),    "paint": (0, 50)},
        "Moderate": {"labor": (75, 150),    "parts": (100, 300),   "paint": (50, 100)},
        "Severe":   {"labor": (100, 200),   "parts": (150, 500),   "paint": (50, 100)},
    },
    "headlight": {
        "Minor":    {"labor": (50, 100),    "parts": (100, 300),   "paint": (0, 0)},
        "Moderate": {"labor": (75, 150),    "parts": (200, 600),   "paint": (0, 0)},
        "Severe":   {"labor": (100, 200),   "parts": (300, 1200),  "paint": (0, 0)},
    },
    "taillight": {
        "Minor":    {"labor": (50, 80),     "parts": (50, 150),    "paint": (0, 0)},
        "Moderate": {"labor": (75, 120),    "parts": (100, 350),   "paint": (0, 0)},
        "Severe":   {"labor": (100, 180),   "parts": (150, 600),   "paint": (0, 0)},
    },
    "rocker_panel": {
        "Minor":    {"labor": (200, 400),   "parts": (100, 250),   "paint": (150, 300)},
        "Moderate": {"labor": (400, 800),   "parts": (250, 600),   "paint": (300, 450)},
        "Severe":   {"labor": (700, 1400),  "parts": (400, 1000),  "paint": (400, 600)},
    },
    "generic_panel": {
        "Minor":    {"labor": (150, 350),   "parts": (50, 200),    "paint": (100, 250)},
        "Moderate": {"labor": (350, 700),   "parts": (200, 600),   "paint": (250, 400)},
        "Severe":   {"labor": (600, 1200),  "parts": (400, 1200),  "paint": (350, 550)},
    },
}


def _map_location_to_panel(location: str) -> str:
    """
    Map a spatial location string from the damage analyzer to the most
    likely vehicle panel. Uses heuristic mapping based on typical vehicle
    photo compositions.

    In a real-time image: front of the car is usually center/right,
    rear is center/left, top is roof/windshield, bottom is bumpers/rockers.
    """
    loc = location.lower().strip()

    if "upper" in loc and ("center" in loc or "right" in loc):
        return "windshield"
    elif "upper" in loc and "left" in loc:
        return "roof"
    elif "lower" in loc and ("left" in loc or "right" in loc):
        return "rocker_panel"
    elif "lower" in loc and "center" in loc:
        return "front_bumper"
    elif "center" in loc and loc == "center":
        return "door"
    elif "left" in loc and "center" not in loc:
        return "quarter_panel"
    elif "right" in loc and "center" not in loc:
        return "quarter_panel"
    else:
        return "generic_panel"


def _estimate_single_region(
    location: str,
    severity: str,
    area_pct: float,
    panel_override: Optional[str] = None,
) -> Dict:
    """
    Estimate repair cost for a single damage region.

    Args:
        location: Spatial location from damage analyzer.
        severity: 'Minor', 'Moderate', or 'Severe'.
        area_pct: Damage area as percentage of image.
        panel_override: Explicit panel name (bypasses location mapping).

    Returns:
        Dict with panel, severity, cost ranges, and midpoint estimate.
    """
    panel = panel_override or _map_location_to_panel(location)
    severity = severity.capitalize()
    if severity not in ("Minor", "Moderate", "Severe"):
        severity = "Moderate"

    costs = _REPAIR_COSTS.get(panel, _REPAIR_COSTS["generic_panel"])
    severity_costs = costs.get(severity, costs["Moderate"])

    labor_low, labor_high = severity_costs["labor"]
    parts_low, parts_high = severity_costs["parts"]
    paint_low, paint_high = severity_costs["paint"]

    total_low = labor_low + parts_low + paint_low
    total_high = labor_high + parts_high + paint_high
    midpoint = (total_low + total_high) // 2

    # Scale up slightly for large damage areas
    if area_pct > 15:
        scale = 1.3
    elif area_pct > 8:
        scale = 1.15
    else:
        scale = 1.0

    return {
        "panel": panel.replace("_", " ").title(),
        "severity": severity,
        "area_pct": area_pct,
        "labor_range": (int(labor_low * scale), int(labor_high * scale)),
        "parts_range": (int(parts_low * scale), int(parts_high * scale)),
        "paint_range": (int(paint_low * scale), int(paint_high * scale)),
        "total_range": (int(total_low * scale), int(total_high * scale)),
        "midpoint_estimate": int(midpoint * scale),
    }


def estimate_repair_costs(damage_analyses: List[Dict]) -> Dict:
    """
    Generate a full repair cost estimate from damage analysis results.

    Args:
        damage_analyses: List of analysis dicts from damage_analyzer.analyze_masks().

    Returns:
        Dict containing per-region estimates, totals, and formatted summary.
    """
    all_estimates = []
    total_low = 0
    total_high = 0

    for analysis in damage_analyses:
        if not analysis.get("damage_detected"):
            continue

        for region in analysis.get("regions", []):
            estimate = _estimate_single_region(
                location=region.get("location", "center"),
                severity=region.get("severity", "Moderate"),
                area_pct=region.get("area_pct", 1.0),
            )
            estimate["image_index"] = analysis.get("image_index", 0)
            estimate["region_id"] = region.get("region_id", 0)
            all_estimates.append(estimate)

            total_low += estimate["total_range"][0]
            total_high += estimate["total_range"][1]

    total_midpoint = (total_low + total_high) // 2

    return {
        "estimates": all_estimates,
        "total_low": total_low,
        "total_high": total_high,
        "total_midpoint": total_midpoint,
        "num_regions": len(all_estimates),
    }


def format_repair_estimate(result: Dict, deductible: int = 0) -> str:
    """
    Format repair estimate results into a readable string for display
    and LLM context.

    Args:
        result: Output from estimate_repair_costs().
        deductible: Policy deductible amount.

    Returns:
        Formatted estimate text.
    """
    if not result["estimates"]:
        return "No repair estimates available (no damage detected)."

    lines = []
    lines.append("REPAIR COST ESTIMATE")
    lines.append("=" * 55)
    lines.append("Note: Estimates based on industry averages. Actual costs")
    lines.append("may vary by region, vehicle make/model, and shop rates.")
    lines.append("")

    for est in result["estimates"]:
        lines.append(f"Image {est['image_index']}, Region {est['region_id']}: {est['panel']}")
        lines.append(f"  Severity: {est['severity']} ({est['area_pct']}% of image)")
        lines.append(f"  Labor:  ${est['labor_range'][0]:,} - ${est['labor_range'][1]:,}")
        lines.append(f"  Parts:  ${est['parts_range'][0]:,} - ${est['parts_range'][1]:,}")
        if est['paint_range'][1] > 0:
            lines.append(f"  Paint:  ${est['paint_range'][0]:,} - ${est['paint_range'][1]:,}")
        lines.append(f"  Subtotal: ${est['total_range'][0]:,} - ${est['total_range'][1]:,}")
        lines.append("")

    lines.append("-" * 55)
    lines.append(f"TOTAL ESTIMATED REPAIR COST: ${result['total_low']:,} - ${result['total_high']:,}")
    lines.append(f"MIDPOINT ESTIMATE: ${result['total_midpoint']:,}")

    if deductible > 0:
        out_of_pocket_low = min(deductible, result["total_low"])
        out_of_pocket_high = min(deductible, result["total_high"])
        insurer_pays_low = max(0, result["total_low"] - deductible)
        insurer_pays_high = max(0, result["total_high"] - deductible)
        lines.append("")
        lines.append(f"DEDUCTIBLE: ${deductible:,}")
        lines.append(f"YOUR OUT-OF-POCKET: ${out_of_pocket_low:,} - ${out_of_pocket_high:,}")
        lines.append(f"INSURER PAYS (estimated): ${insurer_pays_low:,} - ${insurer_pays_high:,}")

    return "\n".join(lines)


def get_estimate_summary_for_llm(
    damage_analyses: List[Dict],
    deductible: int = 0,
) -> Tuple[str, Dict]:
    """
    Convenience function: runs the full estimate pipeline and returns
    both the formatted text (for LLM context) and the raw result dict
    (for UI rendering).

    Args:
        damage_analyses: From damage_analyzer.analyze_masks().
        deductible: Policy deductible.

    Returns:
        Tuple of (formatted_text, raw_result_dict).
    """
    result = estimate_repair_costs(damage_analyses)
    text = format_repair_estimate(result, deductible=deductible)
    return text, result