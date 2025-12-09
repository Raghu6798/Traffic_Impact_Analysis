import os
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
import numpy as np 
import pandas as pd 

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


if __name__ == "__main__":
    result = generate_traffic_demand(
        corridor_mapping_file="final_mapping.json",
        net_file_path="map.net.xml",
        csv_data_path=r"C:\Users\Raghu\Downloads\Traffic_Analysis_Agent\Traffic_Impact_Analysis\docs\Intersection_2.csv",
        flow_file="flows.xml",
        turn_file="turns.xml"
    )
    print(result)