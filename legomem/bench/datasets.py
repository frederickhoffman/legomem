import json
import os
from typing import Any


class OfficeBenchLoader:
    def __init__(self, data_dir: str = "data/officebench"):
        self.data_dir = data_dir

    def load_level(self, level: int) -> list[dict[str, Any]]:
        file_path = os.path.join(self.data_dir, f"level_{level}.json")
        if not os.path.exists(file_path):
            # Return some mock data if file doesn't exist for demo
            return [
                {
                    "id": f"L{level}-1",
                    "description": f"Mock level {level} task: Update the report and notify the team.",
                    "expected_output": "Success"
                }
            ]
        
        with open(file_path) as f:
            return json.load(f)

    def load_all_levels(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "Level 1": self.load_level(1),
            "Level 2": self.load_level(2),
            "Level 3": self.load_level(3)
        }
