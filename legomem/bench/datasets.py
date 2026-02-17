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
                    "description": "Schedule a weekly sync on Bob's calendar for every Friday at 4 PM.",
                    "expected_output": "Weekly sync scheduled for Bob on Fridays at 4 PM."
                }
            ]
        elif level == 2:
            return [
                {
                    "id": "L2-1",
                    "description": "Cross-reference the budget spreadsheet with the Q3 report.",
                    "expected_output": "Budget cross-referenced with report."
                }
            ]
        return []

    def load_all_levels(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "Level 1": self.load_level(1),
            "Level 2": self.load_level(2)
        }
