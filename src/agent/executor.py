from src.agent.tools import (
    execute_shell_commands,
    rename_file,
    download_osm_map,
    extract_candidate_junctions,
    map_volume_to_topology,
    generate_traffic_demand,
    create_sumo_config,
    analyze_simulation_results,
    parse_tripinfo,
    generate_detectors,
    export_simulation_files,
    compute_hcm_metrics,
    parse_queue_xml
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.agents import create_agent

from src.agent.prompts import SYSTEM_PROMPT
from src.config.settings import get_settings

settings = get_settings()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=settings.GOOGLE_API_KEY)


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
        generate_detectors,
        export_simulation_files,
        compute_hcm_metrics,
        parse_queue_xml],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver()
) 