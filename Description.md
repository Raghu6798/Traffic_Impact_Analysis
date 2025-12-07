TRAFFIC IMPACT ANALYSIS (TIA) AGENT - CURRENT CAPABILITIES
Status: Phase 1 (Network) & Phase 2 (Topology) Complete

1. AUTOMATED MAP PREPARATION
   - Downloads raw OpenStreetMap (OSM) data based on user-defined coordinates.
   - Converts OSM data into SUMO Network format (.net.xml) using 'netconvert'.
   - Applies specific TIA cleaning flags (geometry removal, junction joining, traffic light guessing) to ensure simulation stability.
   - Forces preservation of Street Names within the network file for later data matching.

2. DATA DISCOVERY & BATCH MANAGEMENT
   - Parses a "Master Catalog" (CSV/Excel) to handle multi-intersection corridor studies.
   - Identifies individual traffic volume data files associated with specific intersections.
   - Extracts target street names (e.g., "Woodmen Rd", "Meridian") dynamically from the dataset headers.

3. NETWORK TOPOLOGY SCANNING
   - Scans the generated SUMO map to identify valid intersection candidates.
   - Filters out internal geometry nodes and simple road curves to focus on relevant junctions.
   - Caches huge network datasets into local JSON files to prevent LLM context overflow.

4. SEMANTIC-GEOMETRIC MAPPING (The "Bridge")
   - Links "Human" data to "Simulation" data.
   - Matches ambiguous street names from Excel (e.g., "Foxtail Mdw") to precise SUMO Edge IDs.
   - Uses geometric math (vector angles) to automatically determine Cardinal Directions (NB, SB, EB, WB) for every approach edge.
   - Generates a precise "Corridor Mapping" JSON that tells the system exactly which map edge corresponds to which column in the traffic data.

CURRENT STATE:
The Agent now possesses a fully mapped skeleton of the simulation. It knows WHERE every intersection is and WHICH edges represent the North/South/East/West approaches. It is ready to receive traffic volumes.


testing coordinates : North 38.94185,West -104.62124 , East -104.59881 , South 38.93259 


docker run -it -v C:\Users\Raghu\Downloads\Traffic_Analysis_Agent\docs:/app/docs -e GOOGLE_API_KEY="**********************************************" -e HEADLESS_MODE="true" tia-agent

docker run -it --entrypoint /bin/bash -v C:\Users\Raghu\Downloads\Traffic_Analysis_Agent\docs:/app/docs -e GOOGLE_API_KEY="********************************" -e HEADLESS_MODE="true" tia-agent

 North 38.94185,West -104.62124 , East -104.59881 , South 38.93259 are the rectangular box coordinates and the region of interest , traffic volume data at the intersection of Meridian and Woodman road, is /app/docs/Intersection_2.csv ,  run the simulation using sumo command