import sys
import xml.etree.ElementTree as ET

def generate_coverage_badge(xml_path, output_path):
    """
    Parses a coverage.xml file, calculates the coverage percentage, and 
    generates an SVG badge file.
    """
    try:
        # 1. Parse the coverage.xml file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 2. Extract coverage percentage
        # Coverage percentage is stored as a string in 'line-rate' attribute of the <coverage> tag
        coverage_rate_str = root.get('line-rate', '0')
        
        # Convert rate (0.0 to 1.0) to percentage (0 to 100)
        coverage_percent = int(float(coverage_rate_str) * 100)
        
    except FileNotFoundError:
        print(f"Error: coverage XML file not found at '{xml_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing coverage XML: {e}")
        sys.exit(1)

    # 3. Determine color based on threshold
    if coverage_percent >= 90:
        color = "#4c1" # Bright green
    elif coverage_percent >= 80:
        color = "#a4a61d" # Yellow-green
    elif coverage_percent >= 60:
        color = "#dfb317" # Yellow
    else:
        color = "#e05d44" # Red

    # 4. Define the SVG template
    # This uses a standard shields.io style SVG structure
    svg_template = f"""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="100" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h50v20H0z"/>
    <path fill="{color}" d="M50 0h50v20H50z"/>
    <path fill="url(#b)" d="M0 0h100v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11px">
    <text x="25" y="15" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="430">coverage</text>
    <text x="25" y="14" transform="scale(.1)" textLength="430">coverage</text>
    <text x="75" y="15" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="430">{coverage_percent}%</text>
    <text x="75" y="14" transform="scale(.1)" textLength="430">{coverage_percent}%</text>
  </g>
</svg>"""

    # 5. Write the SVG file
    with open(output_path, 'w') as f:
        f.write(svg_template)
    
    print(f"Successfully generated coverage badge: {output_path} ({coverage_percent}%)")


if __name__ == '__main__':
    # Expects two arguments: the path to the coverage.xml and the output SVG path
    if len(sys.argv) != 3:
        print("Usage: python generate_coverage_badge.py <input_coverage_xml_path> <output_badge_svg_path>")
        sys.exit(1)

    xml_file = sys.argv[1]
    svg_file = sys.argv[2]
    
    generate_coverage_badge(xml_file, svg_file)
