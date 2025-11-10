import sys
import math
import xml.etree.ElementTree as ET
from typing import Tuple


def _extract_line_rate(root: ET.Element) -> float:
    """
    Try to extract a line coverage rate (0.0 - 1.0) from a Cobertura style coverage.xml root element.
    Prefers 'line-rate' attribute; falls back to lines-covered / lines-valid if present.
    Returns 0.0 if nothing usable found.
    """
    # Primary attribute
    line_rate = root.get("line-rate")
    if line_rate:
        try:
            return max(0.0, min(1.0, float(line_rate)))
        except (ValueError, TypeError):
            pass

    # Fallback (Cobertura often includes these)
    covered = root.get("lines-covered")
    valid = root.get("lines-valid")
    try:
        if covered is not None and valid is not None:
            covered_i = int(covered)
            valid_i = int(valid)
            if valid_i > 0:
                return max(0.0, min(1.0, covered_i / valid_i))
    except (ValueError, TypeError):
        pass

    return 0.0


def _format_percentage(pct: float) -> Tuple[str, int]:
    """
    Format percentage for display. Keeps one decimal if not an integer.
    Returns (display_str, rounded_int_for_thresholds)
    """
    pct_value = pct * 100.0
    if math.isclose(pct_value, round(pct_value)):
        display = f"{int(round(pct_value))}%"
        rounded = int(round(pct_value))
    else:
        display = f"{pct_value:.1f}%"
        rounded = int(round(pct_value))
    return display, rounded


def _color_for_coverage(pct_int: int) -> str:
    """
    Derive color similar to shields.io thresholds.
    """
    if pct_int >= 90:
        return "#4c1"      # bright green
    if pct_int >= 80:
        return "#97CA00"   # greenish
    if pct_int >= 70:
        return "#a4a61d"   # olive / yellow-green
    if pct_int >= 60:
        return "#dfb317"   # yellow
    if pct_int >= 50:
        return "#fe7d37"   # orange
    return "#e05d44"       # red


def _compute_widths(left_text: str, right_text: str) -> Tuple[int, int, int]:
    """
    Roughly estimate text box widths. Shields approximates ~6-7px per char at 11px font.
    Add horizontal padding each side.
    """
    char_width = 6.5
    padding = 10  # total horizontal padding per box

    left_w = int(char_width * len(left_text) + padding)
    right_w = int(char_width * len(right_text) + padding)

    # Ensure minimum widths for aesthetics
    left_w = max(left_w, 55)
    right_w = max(right_w, 40)

    total = left_w + right_w
    return left_w, right_w, total


def generate_coverage_badge(xml_path: str, output_path: str) -> None:
    """
    Parse a coverage.xml file (Cobertura style), compute coverage %, and generate an SVG badge.
    Improves layout by:
      - Dynamic badge width based on text lengths
      - Proper vertical text alignment (no scaling hack)
      - Robust coverage extraction & formatting
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: coverage XML file not found at '{xml_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing coverage XML: {e}")
        sys.exit(1)

    line_rate = _extract_line_rate(root)
    display_pct, pct_int = _format_percentage(line_rate)
    color = _color_for_coverage(pct_int)

    left_label = "coverage"
    right_value = display_pct
    left_w, right_w, total_w = _compute_widths(left_label, right_value)

    # Vertical alignment: y=14 approximates baseline for 20px height with font-size=11
    # Add a subtle shadow line for readability similar to shields.io
    svg_template = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20" role="img" aria-label="{left_label}: {right_value}">
  <title>{left_label}: {right_value}</title>
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="{total_w}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <rect width="{left_w}" height="20" fill="#555"/>
    <rect x="{left_w}" width="{right_w}" height="20" fill="{color}"/>
    <rect width="{total_w}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{left_w/2:.1f}" y="15" fill="#010101" fill-opacity=".3">{left_label}</text>
    <text x="{left_w/2:.1f}" y="14">{left_label}</text>
    <text x="{left_w + right_w/2:.1f}" y="15" fill="#010101" fill-opacity=".3">{right_value}</text>
    <text x="{left_w + right_w/2:.1f}" y="14">{right_value}</text>
  </g>
</svg>"""

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_template)
    except Exception as e:
        print(f"Error writing SVG badge to '{output_path}': {e}")
        sys.exit(1)

    print(f"Successfully generated coverage badge: {output_path} ({display_pct})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_coverage_badge.py <input_coverage_xml_path> <output_badge_svg_path>")
        sys.exit(1)

    xml_file = sys.argv[1]
    svg_file = sys.argv[2]
    generate_coverage_badge(xml_file, svg_file)
