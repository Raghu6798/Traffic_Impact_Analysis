Filter Rules:

Exclude type="internal": These are hidden nodes used by SUMO to model the exact path inside an intersection box.
Exclude type="dead_end" (Optional but recommended): Usually, you are analyzing intersections where roads cross, not where they end (unless it's a cul-de-sac entry point).
Include Types: "traffic_light", "priority", "right_before_left", "allway_stop".
Edge Count Heuristic: A real intersection usually has at least 3 incoming edges (T-junction) or 4 incoming edges (Cross-junction).



In SUMO's map.net.xml files, the dir attribute within a <connection> element refers to the direction of the connection within an intersection. It specifies which lanes are connected.
While s (straight) is a common value, there are other values that define different turning movements:
s (straight): Indicates a straight-through movement at the intersection.
l (left): Indicates a left turn.
r (right): Indicates a right turn.
t (turn-around): Indicates a U-turn or a turn-around movement.
These direction values are crucial for defining the permissible movements and the flow of traffic within complex intersections, especially when dealing with traffic light logic or priority rules. They help SUMO understand how vehicles can transition between lanes on different edges within a junction.