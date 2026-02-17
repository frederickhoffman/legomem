import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MEMORY_CURATION_PROMPT = """From the following agent trajectory, generate memory that can be useful for future LLM agents’ reference.
# Trajectory: 
{full_trajectory}

# Example:
<start>
{{
  "high_level_plan": "1. Check Bob’s calendar availability for the specified time slot. 2. Add the meeting to Bob’s calendar for 5/17/2024 from 10:30 a.m. to 11:00 a.m.",
  "subtasks": [
    {{
      "agent": "calendar_agent",
      "description": "Check Bob’s schedule on 5/17/2024 from 10:30 a.m. to 11:00 a.m to ensure there are no conflicts",
      "steps": "¡think¿I need to check Bob’s existing calendar events to ensure no scheduling conflicts¡/think¿¡action¿{{\"app\":\"calendar\",\"action\":\"list_events\",\"username\":\"Bob\"}}¡/action¿",
      "observations": "No events found for Bob - calendar is available for the requested time slot"
    }},
    {{
      "agent": "calendar_agent",
      "description": "Add a meeting to Bob’s calendar on 5/17/2024 from 10:30 a.m. to 11:00 a.m",
      "steps": "¡think¿Since no conflicts were found, I can now create the new calendar event for Bob¡/think¿¡action¿{{\"app\":\"calendar\",\"action\":\"create_event\",\"user\":\"Bob\",\"summary\":\"Meeting\",\"time_start\":\"2024-05-17 10:30:00\",\"time_end\":\"2024-05-17 11:00:00\"}}¡/action¿",
      "observations": "Successfully created a new event in Bob’s calendar for the specified date and time"
    }}
  ],
  "final_answer": "The meeting has been successfully added to Bob’s calendar on 5/17/2024 from 10:30 a.m. to 11:00 a.m.",
  "reflections": "Task completed successfully without any conflicts or errors. The calendar check confirmed availability, and the meeting was created with proper date/time formatting."
}}
<end>

# Instructions:
Please analyze the trajectory and extract structured memory with clear thinking and well-formed actions. Use the following format for each subtask step: ¡think¿reasoning¡/think¿ ¡action¿precise tool call¡/action¿

# Rules to follow:
1. Group together actions into subtasks if they are related.
2. Use think-action format for each step.
3. Remove function call IDs but keep tool call structure.
4. Only include successful actions.
5. Keep observations concise.
6. Do not include orchestrator coordination steps.
7. Use string format for the steps field.
Follow JSON format exactly between <start> and <end>.
"""

class MemoryCurator:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def curate_trajectory(self, trajectory: str) -> dict[str, Any]:
        prompt = MEMORY_CURATION_PROMPT.format(full_trajectory=trajectory)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content:
            try:
                # Extract content between <start> and <end>
                start_tag = "<start>"
                end_tag = "<end>"
                start_idx = content.find(start_tag) + len(start_tag)
                end_idx = content.find(end_tag)
                json_str = content[start_idx:end_idx].strip()
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing curated memory: {e}")
                print(f"Raw content: {content}")
        return {}

    def extract_subtasks(self, curated_memory: dict[str, Any]) -> list[dict[str, Any]]:
        return curated_memory.get("subtasks", [])
