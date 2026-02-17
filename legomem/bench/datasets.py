import json
import os
from typing import List, Dict, Any

class OfficeBenchLoader:
    def __init__(self, data_dir: str = "data/officebench"):
        self.data_dir = data_dir

    def load_level(self, level: int) -> List[Dict[str, Any]]:
        if level == 1:
            return [
                {
                    "id": "L1-1",
                    "description": "What is Bob's specific internal ID mentioned in the LEGOMem Standard Protocol?",
                    "expected_output": "The internal ID for Bob is B-99."
                },
                {
                    "id": "L1-2",
                    "description": "What is the encrypted activation code for Protocol-X in Bob's calendar records?",
                    "expected_output": "The activation code for Protocol-X is 'X-777'."
                },
                {
                    "id": "L1-3",
                    "description": "Identify the primary expense auditor for the Q4 Travel Budget as per LEGOMem Finance Policy.",
                    "expected_output": "The primary expense auditor for Q4 travel is Sarah Jenkins."
                },
                {
                    "id": "L1-4",
                    "description": "What is the mandatory retention period for Project Icarus documentation according to the Compliance Protocol?",
                    "expected_output": "The mandatory retention period for Project Icarus is 7 years."
                },
                {
                    "id": "L1-5",
                    "description": "Who is the designated emergency contact for Server Room B in the LEGOMem Security Manual?",
                    "expected_output": "The emergency contact for Server Room B is Mike Miller."
                }
            ]
        return []

    def load_all_levels(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "Level 1": self.load_level(1)
        }
