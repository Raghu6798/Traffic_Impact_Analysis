import json,sys
from loguru import logger
import xml.etree.ElementTree as ET
import math
from langchain_core.tools import tool


logger.remove()

logger.add(
    sys.stdout, 
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> - <level>{message}</level>"
)

def extract_candidate_junctions(net_file_path: str = "map.net.xml"):
    """
    Scans the map for valid intersections and extracts STREET NAMES 
    to allow matching with the Traffic Count Excel file.
    """
    logger.info(f"🔎 Scanning {net_file_path} for intersections with names...")
    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()
        
        # 1. Build a lookup of Edge ID -> Street Name
        # SUMO stores names in the 'name' attribute OR in a <param key="name"/> child
        edge_names = {}
        for edge in root.findall('edge'):
            e_id = edge.get('id')
            
            # Check attribute 'name' (common in recent SUMO versions)
            name = edge.get('name')
            
            # If not in attribute, check <param> children
            if not name:
                for param in edge.findall('param'):
                    if param.get('key') == 'name':
                        name = param.get('value')
                        break
            
            # Fallback: if no name, use ID
            edge_names[e_id] = name if name else e_id

        # 2. Find Candidates
        candidates = []
        for junction in root.findall('junction'):
            j_id = junction.get('id')
            j_type = junction.get('type')
            
            # Filter criteria (Page 3)
            if j_type in ["internal", "dead_end"]:
                continue
            
            inc_lanes = junction.get('incLanes', '').split()
            incoming_edges_set = set()
            incoming_details = []

            for lane in inc_lanes:
                if lane.startswith(":"): continue # Skip internal lanes
                
                # Get Edge ID from Lane ID (remove last _0)
                edge_id = "_".join(lane.split("_")[:-1])
                
                if edge_id not in incoming_edges_set:
                    incoming_edges_set.add(edge_id)
                    # Add the name to the list
                    incoming_details.append({
                        "edge_id": edge_id,
                        "name": edge_names.get(edge_id, "Unknown")
                    })

            # Filter: Real intersections usually have 3+ legs, or are signals
            if len(incoming_edges_set) >= 3 or j_type == "traffic_light":
                candidates.append({
                    "junction_id": j_id,
                    "type": j_type,
                    "num_legs": len(incoming_edges_set),
                    "approaches": incoming_details # Now contains Names!
                })

        # Sort by complexity
        candidates.sort(key=lambda x: x['num_legs'], reverse=True)
        with open('candidates.json', 'w') as f:
            json.dump(candidates, f, indent=2)
        return json.dumps(candidates, indent=2)

    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {e}"

def map_volume_to_topology(net_file: str, candidates_json_path: str, target_streets: str):
    """
    Matches the specific intersection from the Excel data to the Map candidates.
    Determines Edge IDs for NB, SB, EB, WB and SAVES the result to 'final_mapping.json'.
    
    Args:
        net_file: Path to map.net.xml
        candidates_json_path: Path to the JSON file containing the candidates.
        target_streets: Comma-separated street names (e.g. "Woodmen , Meridian") , do not add Rd after the names
    """
    
    logger.info(f"🧭 Mapping volume data to topology for: {target_streets}")
    
    targets = [t.strip().lower() for t in target_streets.split(",")]
    
    try:
        with open(candidates_json_path, 'r') as f:
            candidates = json.load(f)
    except Exception as e:
        return f"Error reading candidates file: {e}"
    
    # 1. FIND THE BEST MATCH JUNCTION
    best_junction = None
    max_matches = 0
    
    for cand in candidates:
        match_count = 0
        cand_names = [app['name'].lower() for app in cand['approaches']]
        
        # Check if target streets appear in this junction's approaches
        for target in targets:
            if any(target in name for name in cand_names):
                match_count += 1
        
        if match_count > max_matches:
            max_matches = match_count
            best_junction = cand
            
    if not best_junction or max_matches == 0:
        logger.error(f"Error: Could not find a junction matching those street names.")
        return "Error: Could not find a junction matching those street names."
        
    logger.success(f"Matched Intersection: {best_junction['junction_id']} with {max_matches} street matches.")

    # 2. DETERMINE CARDINAL DIRECTIONS (Geometry)
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    mapping = {} 
    
    for approach in best_junction['approaches']:
        edge_id = approach['edge_id']
        edge_xml = root.find(f".//edge[@id='{edge_id}']")
        
        if edge_xml is None: continue

        lane = edge_xml.find('lane')
        if lane is None: continue

        shape_str = lane.get('shape')
        if not shape_str: continue

        coords = [list(map(float, p.split(','))) for p in shape_str.split()]
        if len(coords) < 2: continue
        
        # Calculate angle of the LAST segment (The part touching the junction)
        p1 = coords[-2]
        p2 = coords[-1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Determine Direction
        direction = "Unknown"
        if -45 <= angle_deg <= 45: direction = "EB"
        elif 45 < angle_deg <= 135: direction = "NB"
        elif -135 <= angle_deg < -45: direction = "SB"
        else: direction = "WB"
        
        if direction not in mapping:
            mapping[direction] = {"in_edge": edge_id, "name": approach['name']}

    # 3. CONSTRUCT AND SAVE OUTPUT
    
    # Prepare the dictionary object first
    final_output = {
        "junction_id": best_junction['junction_id'],
        "mapping": mapping
    }
    
    # Write to file
    output_filename = "final_mapping.json"
    try:
        with open(output_filename, "w") as f:
            json.dump(final_output, f, indent=2)
        logger.info(f"💾 Result saved to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to write file: {e}")

    # Return the JSON string to the Agent so it knows what happened
    return json.dumps(final_output, indent=2)

if __name__ == "__main__":
    extract_candidate_junctions()
    results = map_volume_to_topology("map.net.xml", "candidates.json", "Meridian,Flower")
    print(results)