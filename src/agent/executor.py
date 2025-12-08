from agent.tools import (
    execute_shell_commands,
    rename_file,
    download_osm_map,
    extract_candidate_junctions,
    map_volume_to_topology,
    generate_traffic_demand,
    create_sumo_config,
    analyze_simulation_results,
    parse_tripinfo,
    generate_detectors
)
from agent.prompts import SYSTEM_PROMPT

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
) c 