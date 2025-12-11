import os
import pandas as pd 
import numpy as np
import sys
import json 
import subprocess
import math
import requests
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from langchain_core.tools import tool

# Local imports
from src.utils.sumo_utils import get_sumo_binary
from src.utils.logger import logger

# --- HELPER FUNCTION FOR LAMBDA PERMISSIONS ---
def _enforce_tmp_path(file_path: str) -> str:
    """
    AWS Lambda Helper: valid writes can ONLY happen in /tmp.
    This function forces any file path to be rooted in /tmp/.
    """
    if not file_path:
        return "/tmp/temp_output.xml"
    
    # If it is already a valid /tmp path, return it
    if file_path.startswith("/tmp/") or file_path.startswith("\\tmp\\"):
        return file_path
        
    # Otherwise, strip directory and force /tmp
    filename = os.path.basename(file_path)
    safe_path = os.path.join("/tmp", filename)
    
    # Only log if we are actually changing the path
    if file_path != safe_path:
        logger.info(f"🔒 Path Sanitization: Redirecting '{file_path}' to '{safe_path}'")
        
    return safe_path

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

@tool
def read_file_head(file_path: str):
    """Reads a CSV or XLSX file and returns its contents as text."""
    try:
        # Check if file exists in the provided path, otherwise check /tmp
        if not os.path.exists(file_path):
            tmp_path = os.path.join("/tmp", os.path.basename(file_path))
            if os.path.exists(tmp_path):
                file_path = tmp_path
            else:
                return f"Error: File {file_path} not found."

        logger.info(f"Reading the file: {file_path}")

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            return "Unsupported file format. Only CSV and XLSX are supported."

        content = df.to_string(index=False)
        return content

    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {e}"

@tool 
def parse_tripinfo(tripinfo_file_path: str="/tmp/tripinfo.xml"):
    """
    Parses a SUMO tripinfo.xml file and returns a summary of the trips.
    """
    tripinfo_file_path = _enforce_tmp_path(tripinfo_file_path)
    if not os.path.exists(tripinfo_file_path):
        return f"Error: File not found at {tripinfo_file_path}"

    try:
        tree = ET.parse(tripinfo_file_path)
        root = tree.getroot()
        
        data = []
        for child in root.findall('tripinfo'):
            trip_id = child.get('id')
            duration = float(child.get('duration', 0))
            distance = float(child.get('routeLength', 0))
            time_loss = float(child.get('timeLoss', 0))
            
            speed = distance / duration if duration > 0 else 0
            
            data.append({
                "trip_id": trip_id,
                "duration_sec": duration,
                "distance_m": distance,
                "time_loss_sec": time_loss,
                "speed_mps": speed
            })
            
        df = pd.DataFrame(data)
        
        if df.empty:
            return "No trips found in tripinfo.xml."
            
        summary = f"Parsed {len(df)} trips.\n"
        summary += f"Avg Duration: {df['duration_sec'].mean():.2f} s\n"
        summary += f"Avg Time Loss (Delay): {df['time_loss_sec'].mean():.2f} s\n"
        summary += f"Avg Speed: {df['speed_mps'].mean():.2f} m/s"
        
        return summary

    except Exception as e:
        logger.error(f"Error parsing tripinfo: {e}")
        return f"Error parsing tripinfo: {e}"

@tool
def create_sumo_config(net_file: str = "/tmp/map.net.xml", route_file: str = "/tmp/traffic.rou.xml", additional_file: str = "/tmp/traffic_sensors.add.xml"):
    """Creates 'config.sumocfg' for the simulation in /tmp/."""
    net_file = _enforce_tmp_path(net_file)
    route_file = _enforce_tmp_path(route_file)
    additional_file = _enforce_tmp_path(additional_file)
    output_cfg = "/tmp/config.sumocfg"

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
        <tripinfo-output value="/tmp/tripinfo.xml"/>
        <summary-output value="/tmp/summary.xml"/>
    </output>
</configuration>"""
    try:
        with open(output_cfg, "w") as f: f.write(content)
        return f"Success: {output_cfg} created."
    except Exception as e:
        return f"Error creating config: {e}"

@tool
def rename_file(old_name: str, new_name: str):
    """Renames a file. Automatically ensures paths are in /tmp/."""
    try:
        old_name = _enforce_tmp_path(old_name)
        new_name = _enforce_tmp_path(new_name)

        if os.path.exists(old_name):
            if os.path.exists(new_name):
                os.remove(new_name)
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
    flow_file: str = "/tmp/flows.xml", 
    turn_file: str = "/tmp/turns.xml"
):
    """Generates SUMO traffic demand files."""
    corridor_mapping_file = _enforce_tmp_path(corridor_mapping_file)
    net_file_path = _enforce_tmp_path(net_file_path)
    flow_file = _enforce_tmp_path(flow_file)
    turn_file = _enforce_tmp_path(turn_file)
    
    logger.info("🚗 Generating Traffic Demand...")
    
    try:
        with open(corridor_mapping_file, 'r') as f:
            raw_data = json.load(f)
        
        if "junction_id" in raw_data:
            mapping_data = {"Single_Intersection_Run": raw_data}
        else:
            mapping_data = raw_data
            
        tree = ET.parse(net_file_path)
        net_root = tree.getroot()
        
    except Exception as e:
        return f"Critical Error loading inputs: {e}"

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

        # Sanity check for CSV path
        if csv_path and not os.path.exists(csv_path):
             tmp_csv = os.path.join("/tmp", os.path.basename(csv_path))
             if os.path.exists(tmp_csv):
                 csv_path = tmp_csv

        if not csv_path or not os.path.exists(csv_path):
            errors.append(f"ID {int_id}: CSV file not found.")
            continue

        try:
            df = pd.read_csv(csv_path)
            veh_cols = [c for c in df.columns if any(x in c for x in ['Left', 'Thru', 'Right', 'U']) and 'Peds' not in c]
            ped_cols = [c for c in df.columns if 'Peds' in c or 'Ped' in c]

            if not veh_cols:
                errors.append(f"ID {int_id}: No vehicle columns found.")
                continue

            df['Interval_Veh_Total'] = df[veh_cols].sum(axis=1)
            df['Hourly_Rolling'] = df['Interval_Veh_Total'].rolling(window=4).sum().shift(-3)
            peak_idx = df['Hourly_Rolling'].idxmax()
            
            if pd.isna(peak_idx):
                errors.append(f"ID {int_id}: Insufficient data.")
                continue
            
            peak_window = df.iloc[int(peak_idx) : int(peak_idx) + 4]
            peak_veh_sums = peak_window[veh_cols].sum()
            peak_ped_sums = peak_window[ped_cols].sum() if ped_cols else pd.Series()

            for direction in ['NB', 'SB', 'EB', 'WB']:
                if direction not in edge_map: continue
                in_edge_id = edge_map[direction]['in_edge']
                
                v_L = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'Left' in c])
                v_T = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'Thru' in c])
                v_R = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'Right' in c])
                v_U = sum([peak_veh_sums[c] for c in peak_veh_sums.index if direction in c and 'U' in c])
                total_veh = v_L + v_T + v_R + v_U
                
                if total_veh > 0:
                    flow_id = f"flow_{junction_id}_{direction}"
                    flows_xml += f'    <flow id="{flow_id}" begin="0" end="3600" number="{int(total_veh)}" from="{in_edge_id}" type="car"/>\n'

                    probs = {'l': v_L/total_veh, 's': v_T/total_veh, 'r': v_R/total_veh, 't': v_U/total_veh}
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

                p_vol = sum([peak_ped_sums[c] for c in peak_ped_sums.index if direction in c])
                if p_vol > 0:
                    ped_id = f"ped_{junction_id}_{direction}"
                    flows_xml += f'    <personFlow id="{ped_id}" begin="0" end="3600" number="{int(p_vol)}">\n'
                    flows_xml += f'        <walk from="{in_edge_id}" to="{in_edge_id}" arrivalPos="max"/>\n'
                    flows_xml += f'    </personFlow>\n'

            processed_count += 1

        except Exception as e:
            errors.append(f"ID {int_id}: {str(e)}")

    flows_xml += "</routes>"
    turns_xml += "    </interval>\n</turns>"

    try:
        with open(flow_file, "w") as f: f.write(flows_xml)
        with open(turn_file, "w") as f: f.write(turns_xml)
        msg = f"✅ Success: Demand generated. Saved to {flow_file} and {turn_file}."
        if errors: msg += f"\n⚠️ Warnings: {'; '.join(errors)}"
        logger.success(msg)
        return msg
    except Exception as e:
        return f"Error writing output files: {e}"

# --- Corrected Signature: Required argument FIRST ---
@tool
def map_volume_to_topology(
    target_streets: str, 
    net_file: str = "/tmp/map.net.xml", 
    candidates_json_path: str = "/tmp/candidates.json"
):
    """
    Matches intersection from Excel data to Map candidates.
    Args:
        target_streets: Comma-separated list of street names (e.g., "Woodmen, Meridian").
        net_file: Path to map network file.
        candidates_json_path: Path to candidates JSON.
    """
    # Enforce paths
    net_file = _enforce_tmp_path(net_file)
    candidates_json_path = _enforce_tmp_path(candidates_json_path)
    output_filename = "/tmp/final_mapping.json"
    
    logger.info(f"🧭 Mapping volume data to topology for: {target_streets}")
    
    targets = [t.strip().lower() for t in target_streets.split(",")]
    
    try:
        with open(candidates_json_path, 'r') as f:
            candidates = json.load(f)
    except Exception as e:
        return f"Error reading candidates file: {e}"
    
    best_junction = None
    max_matches = 0
    
    for cand in candidates:
        match_count = 0
        cand_names = [app['name'].lower() for app in cand['approaches']]
        for target in targets:
            if any(target in name for name in cand_names):
                match_count += 1
        
        if match_count > max_matches:
            max_matches = match_count
            best_junction = cand
            
    if not best_junction or max_matches == 0:
        logger.error(f"Error: Could not find a junction matching those street names.")
        return "Error: Could not find a junction matching those street names."
        
    logger.success(f"Matched Intersection: {best_junction['junction_id']}")

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
        direction = calculate_direction(shape_str)
        
        if direction != "Unknown" and direction not in mapping:
            mapping[direction] = {"in_edge": edge_id, "name": approach['name']}

    final_output = {
        "junction_id": best_junction['junction_id'],
        "mapping": mapping
    }
    
    try:
        with open(output_filename, "w") as f:
            json.dump(final_output, f, indent=2)
        logger.info(f"💾 Result saved to {output_filename}")
        return json.dumps(final_output, indent=2)
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def extract_candidate_junctions(net_file_path: str = "/tmp/map.net.xml", output_path: str = "/tmp/candidates.json"):
    """
    Scans the map for valid intersections and extracts STREET NAMES.
    Saves to JSON file. Returns Status Message Only.
    """
    net_file_path = _enforce_tmp_path(net_file_path)
    output_path = _enforce_tmp_path(output_path)
    
    logger.info(f"🔎 Scanning {net_file_path}...")
    
    if not os.path.exists(net_file_path):
        return f"Error: Net file {net_file_path} not found."

    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()
        
        edge_names = {}
        for edge in root.findall('edge'):
            e_id = edge.get('id')
            name = edge.get('name')
            if not name:
                for param in edge.findall('param'):
                    if param.get('key') == 'name':
                        name = param.get('value')
                        break
            edge_names[e_id] = name if name else e_id

        candidates = []
        for junction in root.findall('junction'):
            j_id = junction.get('id')
            j_type = junction.get('type')
            
            if j_type in ["internal", "dead_end"]: continue
            
            inc_lanes = junction.get('incLanes', '').split()
            incoming_edges_set = set()
            incoming_details = []

            for lane in inc_lanes:
                if lane.startswith(":"): continue 
                if "_" in lane:
                    edge_id = "_".join(lane.split("_")[:-1])
                else:
                    edge_id = lane
                
                if edge_id not in incoming_edges_set:
                    incoming_edges_set.add(edge_id)
                    incoming_details.append({
                        "edge_id": edge_id,
                        "name": edge_names.get(edge_id, "Unknown")
                    })

            if len(incoming_edges_set) >= 3 or j_type == "traffic_light":
                candidates.append({
                    "junction_id": j_id,
                    "type": j_type,
                    "num_legs": len(incoming_edges_set),
                    "approaches": incoming_details
                })

        candidates.sort(key=lambda x: x['num_legs'], reverse=True)
        
        with open(output_path, 'w') as f:
            json.dump(candidates, f, indent=2)
            
        return f"Success: Extracted {len(candidates)} candidates. Saved to {output_path}."

    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {e}"

@tool
def generate_detectors(output_file: str = "/tmp/traffic_sensors.add.xml"):
    """Generates the sensor configuration file."""
    output_file = _enforce_tmp_path(output_file)
    logger.info(f"📡 Generating Traffic Sensors: {output_file}")
    content = """<additional>
    <edgeData id="edge_dump" file="/tmp/edge_performance.xml" freq="900" excludeEmpty="true"/>
    <laneData id="lane_dump" file="/tmp/lane_performance.xml" freq="900" excludeEmpty="true"/>
</additional>
"""
    try:
        with open(output_file, "w") as f:
            f.write(content)
        return f"Success: {output_file} generated."
    except Exception as e:
        return f"Error writing sensors file: {e}"

@tool
def download_osm_map(south: float, west: float, north: float, east: float, filename: str = "/tmp/map.osm"):
    """Downloads a map area from OpenStreetMap."""
    filename = _enforce_tmp_path(filename)

    logger.info(f"🌍 Downloading Map to {filename}")
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
def compute_hcm_metrics(net_file: str = "/tmp/map.net.xml", tripinfo_file: str = "/tmp/tripinfo.xml", queue_file: str = "/tmp/queue.xml"):
    """Computes final TIA/HCM metrics."""
    tripinfo_file = _enforce_tmp_path(tripinfo_file)
    queue_file = _enforce_tmp_path(queue_file)
    results = {"metrics": {}, "details": {}}
    
    if not os.path.exists(tripinfo_file):
        results["metrics"]["error"] = f"Tripinfo file not found at {tripinfo_file}"
        return json.dumps(results)

    try:
        tree = ET.parse(tripinfo_file)
        delays = [float(trip.get('timeLoss')) for trip in tree.getroot().findall('tripinfo')]
        total_vehicles = len(delays)
        avg_delay = sum(delays) / total_vehicles if total_vehicles > 0 else 0
        
        los = "A" if avg_delay <= 10 else "B" if avg_delay <= 20 else "C" if avg_delay <= 35 else "D" if avg_delay <= 55 else "E" if avg_delay <= 80 else "F"
        
        results["metrics"]["Average_Delay_sec"] = round(avg_delay, 2)
        results["metrics"]["Level_of_Service"] = los
        results["metrics"]["Total_Vehicles_Processed"] = total_vehicles
        
        ASSUMED_HOURLY_CAPACITY = 14400 
        v_c_ratio = total_vehicles / ASSUMED_HOURLY_CAPACITY
        results["metrics"]["Volume_to_Capacity_Ratio"] = round(v_c_ratio, 3)
        
    except Exception as e:
        results["metrics"]["delay_error"] = str(e)
    
    if os.path.exists(queue_file):
        results["details"]["Queue_status"] = "Queue file found. Processing..."
    else:
        results["details"]["Queue_status"] = "Queue file missing."

    return json.dumps(results, indent=2)

@tool
def parse_queue_xml(queue_file_path: str = "/tmp/queue.xml"):
    """Parses the queue.xml for 95th Percentile Queue."""
    queue_file_path = _enforce_tmp_path(queue_file_path)

    if not os.path.exists(queue_file_path):
        return f"Error: Queue file not found at {queue_file_path}."

    edge_queues = {} 
    
    try:
        tree = ET.parse(queue_file_path)
        root = tree.getroot()

        for data_timestep in root.findall('data'):
            for lane in data_timestep.find('lanes').findall('lane'):
                lane_id = lane.get('id')
                if "_" in lane_id:
                     edge_id = lane_id.rsplit('_', 1)[0]
                else:
                     edge_id = lane_id 

                queue_length_str = lane.get('queueing_length_experimental', lane.get('queueing_length'))
                try:
                    queue_length = float(queue_length_str)
                    if edge_id not in edge_queues: edge_queues[edge_id] = []
                    if queue_length > 0: edge_queues[edge_id].append(queue_length)
                except (ValueError, TypeError):
                    continue

        results_95th = {}
        for edge_id, lengths in edge_queues.items():
            if not lengths:
                results_95th[edge_id] = 0.0
                continue
            percentile_95 = np.percentile(lengths, 95, method='lower') 
            results_95th[edge_id] = round(percentile_95, 2)
        
        return json.dumps(results_95th, indent=2)

    except Exception as e:
        return f"Error parsing queue: {e}"

@tool 
def execute_shell_commands(command:str):
    """Executes a shell command. Auto-detects usage of sumo tools."""
    logger.info(f"🐚 Input Command: {command}")
    
    parts = command.split(" ", 1)
    tool_name = parts[0]

    # Added jtrrouter to the list
    if tool_name in ["netconvert", "sumo", "sumo-gui", "netedit", "jtrrouter"]:
        full_binary = get_sumo_binary(tool_name)
        if len(parts) > 1:
            final_command = f"{full_binary} {parts[1]}"
        else:
            final_command = full_binary
    else:
        final_command = command

    if "sumo-gui" in final_command and os.environ.get("HEADLESS_MODE") == "true":
        logger.info("🐳 Running in Docker: Switching sumo-gui to sumo (headless)")
        final_command = final_command.replace("sumo-gui", "sumo")

    try: 
        result = subprocess.run(final_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if result.stdout: logger.info(f"Stdout: {result.stdout[:50]}...")
        if result.stderr: logger.warning(f"Stderr: {result.stderr[:50]}...")
        return f"STDOUT:\n{result.stdout}\n\nSTDERR (Warnings/Errors):\n{result.stderr}"
    except Exception as e:
        logger.error(str(e))
        return str(e)

@tool
def analyze_simulation_results(tripinfo_file: str = "/tmp/tripinfo.xml"):
    """Reads SUMO tripinfo.xml and calculates LOS."""
    tripinfo_file = _enforce_tmp_path(tripinfo_file)
    if not os.path.exists(tripinfo_file): return f"Error: {tripinfo_file} not found."
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        delays = [float(trip.get('timeLoss')) for trip in root.findall('tripinfo')]
        
        if not delays: return "No vehicles completed the simulation."
        
        avg_delay = sum(delays) / len(delays)
        if avg_delay <= 10: los = "A"
        elif avg_delay <= 20: los = "B"
        elif avg_delay <= 35: los = "C"
        elif avg_delay <= 55: los = "D"
        elif avg_delay <= 80: los = "E"
        else: los = "F"
        
        return f"Simulation Report:\nTotal Vehicles: {len(delays)}\nAverage Delay: {avg_delay:.2f}s\nLevel of Service (LOS): {los}"
    except Exception as e: return f"Error: {e}"