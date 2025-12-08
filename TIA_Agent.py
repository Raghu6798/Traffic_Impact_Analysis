import os
import pandas as pd 
import sys
import json 
import subprocess
import platform
import math
import statistics
import requests
from pydantic import BaseModel 
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

logger.remove()

logger.add(
    sys.stdout, 
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{function}</cyan> - <level>{message}</level>"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
def get_sumo_binary(binary_name: str) -> str:
    """
    Finds the full path to a SUMO tool (netconvert, sumo, etc.) to avoid PATH errors.
    """
    # 1. Check if it's already in the path (e.g., user typed 'netconvert')
    import shutil
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

@tool
def read_file_head(file_path: str):
    """Reads a CSV or XLSX file and returns its contents as text."""
    try:
        logger.info(f"Reading the file: {file_path}")

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format. Only CSV and XLSX are supported."

        content = df.to_string(index=False)

        logger.success("Successfully parsed the file")
        return content

    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {e}"

def calculate_direction(shape_str):
    if not shape_str: return "Unknown"
    coords = [list(map(float, p.split(','))) for p in shape_str.split()]
    if len(coords) < 2: return "Unknown"
    
    # Vector of the last segment entering the junction
    p1, p2 = coords[-2], coords[-1]
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    
    if -45 <= angle <= 45: return "EB"
    elif 45 < angle <= 135: return "NB"
    elif -135 <= angle < -45: return "SB"
    else: return "WB"

@tool 
def parse_tripinfo(tripinfo_file_path: str):
    """
    Parses a SUMO tripinfo.xml file and returns a summary of the trips.
    Extracts duration, distance, and calculated speed.
    """
    if not os.path.exists(tripinfo_file_path):
        return "Error: File not found."

    try:
        tree = ET.parse(tripinfo_file_path)
        root = tree.getroot()
        
        data = []
        for child in root.findall('tripinfo'):
            trip_id = child.get('id')
            depart_lane = child.get('departLane', '')
            arrival_lane = child.get('arrivalLane', '')
            duration = float(child.get('duration', 0))
            distance = float(child.get('routeLength', 0))
            time_loss = float(child.get('timeLoss', 0))
            
            # Extract Edge ID from Lane ID (e.g., "Edge1_0" -> "Edge1")
            from_edge = "_".join(depart_lane.split("_")[:-1])
            to_edge = "_".join(arrival_lane.split("_")[:-1])
            
            speed = distance / duration if duration > 0 else 0
            
            data.append({
                "trip_id": trip_id,
                "from_edge": from_edge,
                "to_edge": to_edge,
                "duration_sec": duration,
                "distance_m": distance,
                "time_loss_sec": time_loss,
                "speed_mps": speed
            })
            
        df = pd.DataFrame(data)
        
        if df.empty:
            return "No trips found in tripinfo.xml."
            
        # Summary Statistics
        summary = f"Parsed {len(df)} trips.\n"
        summary += f"Avg Duration: {df['duration_sec'].mean():.2f} s\n"
        summary += f"Avg Time Loss (Delay): {df['time_loss_sec'].mean():.2f} s\n"
        summary += f"Avg Speed: {df['speed_mps'].mean():.2f} m/s"
        
        return summary

    except Exception as e:
        logger.error(f"Error parsing tripinfo: {e}")
        return f"Error parsing tripinfo: {e}"
    

@tool
def create_sumo_config(net_file: str = "map.net.xml", route_file: str = "traffic.rou.xml", additional_file: str = "traffic_sensors.add.xml"):
    """Creates 'config.sumocfg' for the simulation.
    Args:
        net_file (str): The path to the network file, the map.net.xml file generated by netconvert.
        route_file (str): The path to the route file, the traffic.rou.xml file generated by jtrrouter command.
        additional_file (str): The path to the additional file, the traffic_sensors.add.xml file generated by `generate_detectors()` tool 
    """
    logger.info("⚙️ Creating Config File...")
    content = f"""<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
        <additional-files value="{additional_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <output>
        <tripinfo-output value="tripinfo.xml"/>
        <summary-output value="summary.xml"/>
    </output>
</configuration>"""
    with open("config.sumocfg", "w") as f: f.write(content)
    return "Success: config.sumocfg created."

@tool
def rename_file(old_name: str, new_name: str):
    """
    Renames a file. Use this to save simulation outputs (e.g., rename 'lane_performance.xml' to 'lane_performance_AM.xml')
    before running the next simulation scenario.
    """
    try:
        if os.path.exists(old_name):
            if os.path.exists(new_name):
                os.remove(new_name) # Overwrite if exists
            os.rename(old_name, new_name)
            return f"Success: Renamed {old_name} to {new_name}"
        else:
            return f"Error: Source file {old_name} does not exist."
    except Exception as e:
        return f"Error renaming file: {e}"

@tool
def generate_traffic_demand(
    corridor_mapping_file: str, 
    net_file_path: str,
    csv_data_path: str = None, 
    flow_file: str = "flows.xml", 
    turn_file: str = "turns.xml"
):
    """
    Generates SUMO traffic demand files ('flows.xml' and 'turns.xml').
    Now includes PEDESTRIAN demand and calculates PHF metrics.
    """
    logger.info("🚗 Generating Traffic Demand (Vehicles & Pedestrians)...")
    
    try:
        with open(corridor_mapping_file, 'r') as f:
            raw_data = json.load(f)
        
        # Normalize Input
        if "junction_id" in raw_data:
            mapping_data = {"Single_Intersection_Run": raw_data}
        else:
            mapping_data = raw_data
            
        tree = ET.parse(net_file_path)
        net_root = tree.getroot()
        
    except Exception as e:
        return f"Critical Error loading inputs: {e}"

    # Define vTypes for Car and Pedestrian
    flows_xml = """<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55.55"/>
    <vType id="pedestrian" vClass="pedestrian" length="0.5" width="0.5" speed="1.2"/>\n"""
    
    turns_xml = '<turns>\n    <interval begin="0" end="3600">\n'
    
    processed_count = 0
    errors = []

    for int_id, data in mapping_data.items():
        junction_id = data.get('junction_id')
        edge_map = data.get('mapping', {})
        csv_path = data.get('data_file_path', csv_data_path)

        if not csv_path or not os.path.exists(csv_path):
            errors.append(f"ID {int_id}: CSV file not found.")
            continue

        try:
            df = pd.read_csv(csv_path)
            
            # 1. Identify Columns
            # Vehicle columns
            veh_cols = [c for c in df.columns if any(x in c for x in ['Left', 'Thru', 'Right', 'U']) and 'Peds' not in c]
            # Pedestrian columns (Look for 'Peds' or 'Ped')
            ped_cols = [c for c in df.columns if 'Peds' in c or 'Ped' in c]

            if not veh_cols:
                errors.append(f"ID {int_id}: No vehicle columns found.")
                continue

            # 2. Find Peak Hour (Rolling Sum) - PDF Page 3
            df['Interval_Veh_Total'] = df[veh_cols].sum(axis=1)
            df['Hourly_Rolling'] = df['Interval_Veh_Total'].rolling(window=4).sum().shift(-3)
            
            peak_idx = df['Hourly_Rolling'].idxmax()
            
            if pd.isna(peak_idx):
                errors.append(f"ID {int_id}: Insufficient data.")
                continue
            
            # 3. Calculate PHF (Peak Hour Factor) - PDF Page 4
            # PHF = Hourly Volume / (4 * Max 15-min Volume)
            peak_hour_vol = df.loc[peak_idx, 'Hourly_Rolling']
            peak_window = df.iloc[int(peak_idx) : int(peak_idx) + 4]
            max_15min = peak_window['Interval_Veh_Total'].max()
            
            phf = peak_hour_vol / (4 * max_15min) if max_15min > 0 else 0
            
            logger.info(f"📊 Intersection {junction_id} Stats:")
            logger.info(f"   Peak Hour Starts Index: {peak_idx}")
            logger.info(f"   Total Peak Hourly Volume: {int(peak_hour_vol)}")
            logger.info(f"   Peak Hour Factor (PHF): {phf:.2f}") 

            # Sum volumes for the peak hour
            peak_veh_sums = peak_window[veh_cols].sum()
            peak_ped_sums = peak_window[ped_cols].sum() if ped_cols else pd.Series()

            # 4. Generate Flows per Direction
            for direction in ['NB', 'SB', 'EB', 'WB']:
                if direction not in edge_map: continue
                
                in_edge_id = edge_map[direction]['in_edge']
                
                # --- A. VEHICLES ---
                v_L = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'Left' in c])
                v_T = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'Thru' in c])
                v_R = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'Right' in c])
                v_U = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'U' in c])
                
                total_veh = v_L + v_T + v_R + v_U
                
                if total_veh > 0:
                    # Write Vehicle Flow
                    flow_id = f"flow_{junction_id}_{direction}"
                    flows_xml += f'    <flow id="{flow_id}" begin="0" end="3600" number="{int(total_veh)}" from="{in_edge_id}" type="car"/>\n'

                    # Write Turning Ratios
                    probs = {
                        'l': v_L / total_veh, 's': v_T / total_veh, 
                        'r': v_R / total_veh, 't': v_U / total_veh
                    }
                    
                    connections = net_root.findall(f"./connection[@from='{in_edge_id}']")
                    turns_xml += f'        <fromEdge id="{in_edge_id}">\n'
                    written_dirs = set()
                    
                    for conn in connections:
                        to_edge = conn.get('to')
                        sumo_dir = conn.get('dir', '').lower()
                        if sumo_dir in probs and probs[sumo_dir] > 0 and sumo_dir not in written_dirs:
                            turns_xml += f'            <toEdge id="{to_edge}" probability="{probs[sumo_dir]:.2f}"/>\n'
                            written_dirs.add(sumo_dir)
                    turns_xml += '        </fromEdge>\n'

                # --- B. PEDESTRIANS (PDF Page 4) ---
                # Find columns like "NB_Peds" or "NB Peds"
                p_vol = sum([peak_ped_sums[c] for c in peak_ped_sums.index if direction in c])
                
                if p_vol > 0:
                    # SUMO uses <personFlow> for pedestrians.
                    # We inject them on the walking edge associated with the road.
                    # Simple approach: start them at the edge, let them walk to the junction.
                    ped_id = f"ped_{junction_id}_{direction}"
                    # Note: 'departPos' random distributes them. 'arrivalPos' makes them cross.
                    flows_xml += f'    <personFlow id="{ped_id}" begin="0" end="3600" number="{int(p_vol)}">\n'
                    flows_xml += f'        <walk from="{in_edge_id}" to="{in_edge_id}" arrivalPos="max"/>\n' # Walk to end of edge (intersection)
                    flows_xml += f'    </personFlow>\n'

            processed_count += 1

        except Exception as e:
            errors.append(f"ID {int_id}: {str(e)}")

    flows_xml += "</routes>"
    turns_xml += "    </interval>\n</turns>"

    try:
        with open(flow_file, "w") as f: f.write(flows_xml)
        with open(turn_file, "w") as f: f.write(turns_xml)
        msg = f"✅ Success: Demand generated for {processed_count} intersections."
        if errors: msg += f"\n⚠️ Warnings: {'; '.join(errors)}"
        logger.success(msg)
        return msg
    except Exception as e:
        return f"Error writing output files: {e}"


@tool
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

@tool
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

@tool
def generate_detectors(output_file: str = "traffic_sensors.add.xml"):
    """
    Generates the 'traffic_sensors.add.xml' file.
    This defines Edge and Lane data collectors required for TIA metrics 
    (Volume, Speed, Delay, Density) as per PDF Page 19.
    """
    logger.info(f"📡 Generating Traffic Sensors configuration: {output_file}")
    
    # freq="900" means it aggregates data every 15 minutes (900 seconds)
    # excludeEmpty="true" prevents file bloat by ignoring empty roads
    content = """<additional>
    <!-- Edge-based metrics (Volume, Speed, Density, Travel Time) -->
    <edgeData id="edge_dump" file="edge_performance.xml" freq="900" excludeEmpty="true"/>
    
    <!-- Lane-based metrics (Control Delay per lane) -->
    <laneData id="lane_dump" file="lane_performance.xml" freq="900" excludeEmpty="true"/>
</additional>
"""
    try:
        with open(output_file, "w") as f:
            f.write(content)
        return f"Success: {output_file} generated."
    except Exception as e:
        return f"Error writing sensors file: {e}"

def get_hcm_grade(delay):
    """Returns LOS Letter based on HCM 6th Ed (Signalized) thresholds."""
    if delay <= 10: return "A"
    elif 10 < delay <= 20: return "B"
    elif 20 < delay <= 35: return "C"
    elif 35 < delay <= 55: return "D"
    elif 55 < delay <= 80: return "E"
    elif 80 < delay <= 110: return "F"
    else: 
        return "Can't be graded"


@tool
def download_osm_map(south: float, west: float, north: float, east: float, filename: str = "map.osm"):
    """
    Downloads a map area from OpenStreetMap using the Overpass API.
    Arguments: south, west, north, east (coordinates), and filename (default map.osm).
    """
    logger.info(f"🌍 Downloading Map: S={south}, W={west}, N={north}, E={east}")
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
        (
          node({south},{west},{north},{east});
          way({south},{west},{north},{east});
          rel({south},{west},{north},{east});
        );
        (._;>;);
        out meta;
    """
    try:
        response = requests.post(overpass_url, data=query)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            logger.success(f"Map saved to {filename}")
            return f"Success: Map saved to {filename}"
        else:
            return f"Error: Status Code {response.status_code}"
    except Exception as e:
        return f"Exception: {str(e)}"

@tool 
def execute_shell_commands(command:str):
    """
    Executes a shell command. Use this to run netconvert, sumo, or python scripts.
    IMPORTANT: If using 'netconvert' or 'sumo', the tool will try to auto-resolve the path.
    """
    logger.info(f"🐚 Input Command: {command}")
    
    # --- AUTO-FIX FOR WINDOWS PATHS ---
    parts = command.split(" ", 1)
    tool_name = parts[0]


    # If the user asks for 'netconvert' or 'sumo', resolve the full path
    if tool_name in ["netconvert", "sumo", "sumo-gui", "netedit"]:
        full_binary = get_sumo_binary(tool_name)
        if len(parts) > 1:
            final_command = f"{full_binary} {parts[1]}"
        else:
            final_command = full_binary
        logger.info(f"🔧 Resolved Path: {final_command}")
    else:
        final_command = command

    try: 
        if "sumo-gui" in final_command and os.environ.get("HEADLESS_MODE") == "true":
            logger.info("🐳 Running in Docker: Switching sumo-gui to sumo (headless)")
            final_command = final_command.replace("sumo-gui", "sumo")

        result = subprocess.run(final_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        
        # Log minimal output to keep console clean, return full output to Agent
        if result.stdout: logger.info(f"Stdout (first 50 chars): {result.stdout[:50]}...")
        if result.stderr: logger.warning(f"Stderr (first 50 chars): {result.stderr[:50]}...") # Warnings are common in SUMO
        
        # SUMO writes warnings to stderr, so we return both combined
        return f"STDOUT:\n{result.stdout}\n\nSTDERR (Warnings/Errors):\n{result.stderr}"
        
    except Exception as e:
        logger.error(str(e))
        return str(e)

@tool
def analyze_simulation_results(tripinfo_file: str = "tripinfo.xml"):
    """Reads SUMO tripinfo.xml and calculates Level of Service (LOS)."""
    logger.info("📊 Analyzing Results...")
    if not os.path.exists(tripinfo_file): return "Error: tripinfo.xml not found."
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        delays = [float(trip.get('timeLoss')) for trip in root.findall('tripinfo')]
        
        if not delays: return "No vehicles completed the simulation."
        
        avg_delay = sum(delays) / len(delays)
        
        # LOS thresholds (Signalized)
        if avg_delay <= 10: los = "A"
        elif avg_delay <= 20: los = "B"
        elif avg_delay <= 35: los = "C"
        elif avg_delay <= 55: los = "D"
        elif avg_delay <= 80: los = "E"
        else: los = "F"
        
        return f"Simulation Report:\nTotal Vehicles: {len(delays)}\nAverage Delay: {avg_delay:.2f}s\nLevel of Service (LOS): {los}"
    except Exception as e: return f"Error: {e}"


SYSTEM_PROMPT = """
You are an expert Traffic Simulation Engineer Agent using SUMO on Windows.
Your goal is to build and run a complete **Traffic Impact Analysis (TIA) simulation**.

**EXECUTION PIPELINE:**

**Phase 1: Network & Discovery**
1.  Download Map -> Convert to `map.net.xml`.
    * Use  `execute_shell_commands` tool to run the netconvert command: `netconvert --osm-files map.osm -o map.net.xml --geometry.remove true --junctions.join true --tls.guess true --output.street-names true`
2.  Run `extract_candidate_junctions` (Cache map nodes).

**Phase 2: Topology Alignment**
*   Run `map_volume_to_topology` using the `candidates.json` file.
*   Goal: Map human street names to SUMO Edge IDs.

**Phase 3: Demand Generation**
1.  Call `generate_traffic_demand`.
2.  **CRITICAL:** If handling a single intersection, you MUST pass `csv_data_path` (path to the Excel/CSV volume data).
3.  Output: `flows.xml` and `turns.xml`.

**Phase 4: Routing**
1.  Run `jtrrouter` to create routes.
    *   **MANDATORY COMMAND:**
        `jtrrouter --net-file map.net.xml --route-files flows.xml --turn-ratio-files turns.xml --output-file traffic.rou.xml --begin 0 --end 3600 --accept-all-destinations true --seed 42`
    *   *Note:* The `--accept-all-destinations` flag handles sinks automatically. Do NOT create a sinks.xml file.

**Phase 5: Simulation & Analysis**
1.  Call `create_sumo_config`.
2.  Run Simulation: `execute_shell_commands("sumo-gui -c config.sumocfg")`.
3.  Call `analyze_simulation_results` to report the Level of Service (LOS).
"""
agent = create_agent(
    llm, 
    tools=[execute_shell_commands,
        rename_file,
        download_osm_map,
        extract_candidate_junctions,
        map_volume_to_topology,
        generate_traffic_demand,
        create_sumo_config,
        analyze_simulation_results,
        parse_tripinfo,
        generate_detectors],
    system_prompt=SYSTEM_PROMPT
) 

while True: 
    q=input("👤 User : ")
    if q.lower() == "exit": 
        break 
    response = agent.invoke({"messages":[{"role":"user","content":q}]}) 
    logger.info(f"🤖 Agent : {response['messages'][-1].content}")