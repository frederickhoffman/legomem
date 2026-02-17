import json
import os
from typing import List, Dict, Any

class OfficeBenchLoader:
    def __init__(self, data_dir: str = "data/officebench"):
        self.data_dir = data_dir

    def load_level(self, level: int) -> List[Dict[str, Any]]:
        if level == 1:
            return [
                # --- Procedural Tasks (Memory-Dependent) ---
                {
                    "id": "P-1",
                    "type": "procedural",
                    "description": "What is Bob's specific internal ID mentioned in the LEGOMem Standard Protocol?",
                    "expected_output": "The internal ID for Bob is B-99."
                },
                {
                    "id": "P-2",
                    "type": "procedural",
                    "description": "What is the encrypted activation code for Protocol-X in Bob's calendar records?",
                    "expected_output": "The activation code for Protocol-X is 'X-777'."
                },
                {
                    "id": "P-3",
                    "type": "procedural",
                    "description": "Identify the primary expense auditor for the Q4 Travel Budget as per LEGOMem Finance Policy.",
                    "expected_output": "The primary expense auditor for Q4 travel is Sarah Jenkins."
                },
                {
                    "id": "P-4",
                    "type": "procedural",
                    "description": "What is the mandatory retention period for Project Icarus documentation according to the Compliance Protocol?",
                    "expected_output": "The mandatory retention period for Project Icarus is 7 years."
                },
                {
                    "id": "P-5",
                    "type": "procedural",
                    "description": "Who is the designated emergency contact for Server Room B in the LEGOMem Security Manual?",
                    "expected_output": "The emergency contact for Server Room B is Mike Miller."
                },
                # --- General Reasoning Tasks (Memory-Independent) ---
                {
                    "id": "G-1",
                    "type": "general",
                    "description": "Bob has a meeting at 2 PM and another at 3 PM. Can he schedule a 30-minute call at 2:30 PM? Respond with YES or NO and a brief reason.",
                    "expected_output": "NO. There is a conflict with the 2 PM meeting which likely ends at 3 PM or overlaps."
                },
                {
                    "id": "G-2",
                    "type": "general",
                    "description": "Draft a short, polite email to Alice decling her invitation to the 'Underwater Basket Weaving' workshop due to a prior commitment.",
                    "expected_output": "Email drafted regarding declining the workshop invitation politely."
                },
                {
                    "id": "G-3",
                    "type": "general",
                    "description": "Calculate the total cost: 5 licenses at $100 each and 2 server instances at $500 each.",
                    "expected_output": "The total cost is $1500."
                },
                {
                    "id": "G-4",
                    "type": "general",
                    "description": "Summarize the key point of this sentence: 'Despite the heavy rain and strong winds, the dedicated team managed to complete the construction project ahead of schedule.'",
                    "expected_output": "The team completed the project early despite bad weather."
                },
                {
                    "id": "G-5",
                    "type": "general",
                    "description": "If all inputs are valid, and System A is online, what is the status of the combined system output?",
                    "expected_output": "The status depends on the logic, but generally implies 'Operational' or 'Valid'."
                }
            ]
        return []

    def load_all_levels(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "Level 1": self.load_level(1)
        }
