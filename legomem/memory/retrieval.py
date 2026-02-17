import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .vector_store import VectorStore

load_dotenv()

QUERY_REWRITE_PROMPT = """Based on the following similar task examples, 
break down the new task into a step-by-step plan.
## Similar Task Examples: 
{memory_context}
## New Task: 
{task_description}

Please provide a numbered list of 3-5 high-level steps that would be needed to complete this task.
Focus on the main phases/subtasks, not detailed actions.
Format your response as a simple numbered list enclosed within ¡start¿ and ¡end¿ tags:
¡start¿ 
1. [First step] 
2. [Second step] 
3. [Third step] 
... 
¡end¿
"""

class MemoryRetriever:
    def __init__(self, task_bank: VectorStore, subtask_bank: VectorStore | None = None):
        self.task_bank = task_bank
        self.subtask_bank = subtask_bank
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def retrieve_vanilla(self, task_description: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieves full-task memories at inference time."""
        return self.task_bank.search(task_description, k=k)

    def retrieve_dynamic(self, subtask_description: str, k: int = 3) -> list[dict[str, Any]]:
        """Performs just-in-time, subtask-level retrieval."""
        if not self.subtask_bank:
            return []
        return self.subtask_bank.search(subtask_description, k=k)

    def rewrite_query(
        self, 
        task_description: str, 
        similar_tasks: list[dict[str, Any]]
    ) -> list[str]:
        """Uses an LLM to rewrite the task into subtasks before execution."""
        memory_context = "\n\n".join([
            f"Task: {m.get('task_description', 'N/A')}\nPlan: {m.get('high_level_plan', 'N/A')}"
            for m in similar_tasks
        ])
        
        prompt = QUERY_REWRITE_PROMPT.format(
            memory_context=memory_context,
            task_description=task_description
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        content = response.choices[0].message.content
        if content:
            try:
                start_tag = "¡start¿"
                end_tag = "¡end¿"
                start_idx = content.find(start_tag) + len(start_tag)
                end_idx = content.find(end_tag)
                steps_str = content[start_idx:end_idx].strip()
                return [s.strip() for s in steps_str.split("\n") if s.strip()]
            except Exception as e:
                print(f"Error parsing rewritten query: {e}")
        return []
