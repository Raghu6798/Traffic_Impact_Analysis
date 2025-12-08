import shutil
import os
import math
from pathlib import Path

def get_sumo_binary(binary_name: str) -> str:
    """Finds the full path to a SUMO tool."""
    if shutil.which(binary_name):
        return binary_name

    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo\bin",
        r"C:\Program Files\Eclipse\Sumo\bin",
        os.environ.get("SUMO_HOME", "") + "/bin"
    ]

    for path in possible_paths:
        full_path = Path(path) / f"{binary_name}.exe"
        if full_path.exists():
            return f'"{str(full_path)}"'
    
    return binary_name

def calculate_direction(shape_str):
    if not shape_str: return "Unknown"
    try:
        coords = [list(map(float, p.split(','))) for p in shape_str.split()]
        if len(coords) < 2: return "Unknown"
        
        p1, p2 = coords[-2], coords[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        if -45 <= angle <= 45: return "EB"
        elif 45 < angle <= 135: return "NB"
        elif -135 <= angle < -45: return "SB"
        else: return "WB"
    except Exception:
        return "Unknown"