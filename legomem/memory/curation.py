import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MEMORY_CURATION_PROMPT = (
    "From the following agent trajectory, generate memory "
    "that can be useful for future LLM agents' reference.\n"
    "# Trajectory: \n"
    "{full_trajectory}\n\n"
    "# Example:\n"
    "<start>\n"
    "{{\n"
    "  \"high_level_plan\": \"1. Check Bob's calendar availability. "
    "2. Add the meeting to Bob's calendar.\",\n"
    "  \"subtasks\": [\n"
    "    {{\n"
    "      \"agent\": \"calendar_agent\",\n"
    "      \"description\": \"Check Bob's schedule on 5/17/2024\",\n"
    "      \"steps\": \"¡think¿check calendar¡/think¿¡action¿{{\\\"app\\\":\\\"calendar\\\","
    "\\\"action\\\":\\\"list_events\\\",\\\"username\\\":\\\"Bob\\\"}}¡/action¿\",\n"
    "      \"observations\": \"No events found for Bob\"\n"
    "    }}\n"
    "  ],\n"
    "  \"final_answer\": \"The meeting has been added to Bob's calendar.\",\n"
    "  \"reflections\": \"Task completed successfully.\"\n"
    "}}\n"
    "<end>\n\n"
    "# Instructions:\n"
    "Please analyze the trajectory and extract structured memory.\n"
    "Use format: ¡think¿reasoning¡/think¿ ¡action¿precise tool call¡/action¿\n\n"
    "# Rules:\n"
    "1. Group together actions into subtasks if they are related.\n"
    "2. Use think-action format for each step.\n"
    "3. Remove function call IDs but keep tool call structure.\n"
    "4. Only include successful actions.\n"
    "5. Keep observations concise.\n"
    "6. Do not include orchestrator coordination steps.\n"
    "7. Use string format for the steps field.\n"
    "Follow JSON format exactly between <start> and <end>."
)

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
