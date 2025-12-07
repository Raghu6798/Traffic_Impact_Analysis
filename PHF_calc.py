# import pandas as pd
# import json
# import os
# import xml.etree.ElementTree as ET
# from loguru import logger
# import sys

# # Configure Logger
# logger.remove()
# logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# # --- PASTE THE TOOL CODE HERE ---
# def generate_traffic_demand(corridor_mapping_file: str, net_file_path: str, flow_file: str = "flows.xml", turn_file: str = "turns.xml"):
#     logger.info("🚗 Generating Traffic Demand (Flows & Turns)...")
    
#     try:
#         with open(corridor_mapping_file, 'r') as f:
#             mapping_data = json.load(f)
        
#         tree = ET.parse(net_file_path)
#         net_root = tree.getroot()
        
#     except Exception as e:
#         return f"Critical Error loading inputs: {e}"

#     flows_xml = '<routes>\n    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="55.55"/>\n'
#     turns_xml = '<turns>\n    <interval begin="0" end="3600">\n'
    
#     processed_count = 0
#     errors = []

#     for int_id, data in mapping_data.items():
#         junction_id = data.get('junction_id')
#         csv_path = data.get('data_file_path') 
#         edge_map = data.get('mapping', {})

#         if not csv_path or not os.path.exists(csv_path):
#             errors.append(f"ID {int_id}: CSV file missing.")
#             continue

#         try:
#             # --- STEP 1: CALCULATE PEAK HOUR VOLUMES ---
#             df = pd.read_csv(csv_path)
#             vol_cols = [c for c in df.columns if any(x in c for x in ['Left', 'Thru', 'Right', 'U'])]
            
#             df['Interval_Total'] = df[vol_cols].sum(axis=1)
#             df['Hourly_Rolling'] = df['Interval_Total'].rolling(window=4).sum().shift(-3)
            
#             peak_idx = df['Hourly_Rolling'].idxmax()
#             if pd.isna(peak_idx):
#                 errors.append(f"ID {int_id}: Data insufficient.")
#                 continue
            
#             # Print Stats for Manual Verification
#             logger.info(f"📊 Stats for {csv_path}:")
#             logger.info(f"   Peak Hour Starts at Row Index: {peak_idx}")
#             logger.info(f"   Peak Hour Total Volume: {df.iloc[peak_idx]['Hourly_Rolling']}")

#             peak_slice = df.iloc[int(peak_idx) : int(peak_idx) + 4]
#             peak_sums = peak_slice[vol_cols].sum()

#             # --- STEP 2: GENERATE FLOWS & TURNS ---
#             for direction in ['NB', 'SB', 'EB', 'WB']:
#                 if direction not in edge_map: continue
                
#                 in_edge_id = edge_map[direction]['in_edge']
                
#                 # Fuzzy match column names
#                 v_L = sum([peak_sums[c] for c in peak_sums.index if direction in c and 'Left' in c])
#                 v_T = sum([peak_sums[c] for c in peak_sums.index if direction in c and 'Thru' in c])
#                 v_R = sum([peak_sums[c] for c in peak_sums.index if direction in c and 'Right' in c])
#                 v_U = sum([peak_sums[c] for c in peak_sums.index if direction in c and 'U' in c])
                
#                 total_vol = v_L + v_T + v_R + v_U
#                 if total_vol <= 0: continue

#                 # Write Source Flow
#                 flow_id = f"flow_{junction_id}_{direction}"
#                 flows_xml += f'    <flow id="{flow_id}" begin="0" end="3600" number="{int(total_vol)}" from="{in_edge_id}" type="car"/>\n'

#                 # Calculate Probs
#                 probs = {
#                     'l': v_L / total_vol,
#                     's': v_T / total_vol,
#                     'r': v_R / total_vol,
#                     't': v_U / total_vol
#                 }

#                 # Resolve Connections
#                 connections = net_root.findall(f"./connection[@from='{in_edge_id}']")
                
#                 if not connections:
#                     logger.warning(f"   ⚠️ No connections found in map for edge {in_edge_id}. Turn ratios will be empty.")

#                 turns_xml += f'        <fromEdge id="{in_edge_id}">\n'
#                 written_dirs = set()

#                 for conn in connections:
#                     to_edge = conn.get('to')
#                     sumo_dir = conn.get('dir', '').lower() 
                    
#                     if sumo_dir in probs and probs[sumo_dir] > 0:
#                         if sumo_dir not in written_dirs:
#                             turns_xml += f'            <toEdge id="{to_edge}" probability="{probs[sumo_dir]:.2f}"/>\n'
#                             written_dirs.add(sumo_dir)
                            
#                 turns_xml += '        </fromEdge>\n'
            
#             processed_count += 1

#         except Exception as e:
#             errors.append(f"ID {int_id}: {str(e)}")

#     flows_xml += "</routes>"
#     turns_xml += "    </interval>\n</turns>"

#     with open(flow_file, "w") as f: f.write(flows_xml)
#     with open(turn_file, "w") as f: f.write(turns_xml)
    
#     return f"Done. Processed {processed_count}."

# # --- MANUAL TEST EXECUTION ---
# if __name__ == "__main__":
    
#     # 1. SETUP PATHS
#     # Ensure this points to your real map file
#     NET_FILE = "map.net.xml" 
#     # Ensure this points to one of your real CSVs
#     CSV_FILE = r"C:\Users\Raghu\Downloads\Traffic_Analysis_Agent\docs\Intersection_2.csv" 
#     SINGLE_MAPPING_FILE = r"C:\Users\Raghu\Downloads\Traffic_Analysis_Agent\Traffic_Impact_Analysis\final_mapping.json"
#     try:
#         with open(SINGLE_MAPPING_FILE, 'r') as f:single_data = json.load(f)
            
#         # Inject the CSV path (The tool needs to know where to find volumes)
#         single_data['data_file_path'] = CSV_FILE
        
#         # Create the Batch Structure
#         batch_input = {
#             "1": single_data  # "1" is just a dummy ID
#         }
        
#         # Write this temporary batch file
#         with open("temp_batch_mapping.json", "w") as f:
#             json.dump(batch_input, f, indent=2)
            
#     except Exception as e:
#         print(f"Error preparing input: {e}")
#         exit()
    
    
#     # 3. RUN THE TOOL
#     result = generate_traffic_demand(
#         corridor_mapping_file="temp_batch_mapping.json",
#         net_file_path=NET_FILE
#     )
    
#     print(f"\nResult: {result}")
    
#     # 4. INSPECT OUTPUT
#     print("\n--- 📄 GENERATED FLOWS.XML PREVIEW ---")
#     if os.path.exists("flows.xml"):
#         with open("flows.xml", "r") as f:
#             print(f.read())
            
#     print("\n--- 📄 GENERATED TURNS.XML PREVIEW ---")
#     if os.path.exists("turns.xml"):
#         with open("turns.xml", "r") as f:
#             print(f.read())
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.novita.ai/openai",
    api_key=os.getenv("NOVITA_API_KEY"),
)

response = client.chat.completions.create(
    model="deepseek/deepseek-ocr",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": r"C:\Users\Raghu\OneDrive\Pictures\Screenshots\Screenshot 2025-11-08 005903.png"
            }
          },
          {
            "type": "text",
            "text": "OCR this image."
          }
        ]
      }
    ],
    stream=False,
    max_tokens=4096
)

content = response.choices[0].message.content

print(content)