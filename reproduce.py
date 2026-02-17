import os

from dotenv import load_dotenv

from legomem.eval.evaluator import EvaluationPipeline
from legomem.memory.vector_store import VectorStore

load_dotenv()

def reproduce():
    print("Starting LEGOMem Reproduction...")
    
    # 1. Setup Mock Memories (In a real scenario, these would be from training tasks)
    task_bank_path = "data/memory_bank/task_bank"
    subtask_bank_path = "data/memory_bank/subtask_bank"
    
    os.makedirs("data/memory_bank", exist_ok=True)
    
    task_bank = VectorStore()
    subtask_bank = VectorStore()
    
    # Add a mock successful trajectory memory
    mock_memory = {
        "task_description": (
            "Schedule a meeting with Alice for tomorrow at 2 PM "
            "and send a confirmation email."
        ),
        "high_level_plan": "1. Check calendar availability. 2. Create event. 3. Send email.",
        "subtasks": [
            {"agent": "calendar_agent", "description": "Check availability"},
            {"agent": "calendar_agent", "description": "Create event"},
            {"agent": "email_agent", "description": "Send confirmation"}
        ]
    }
    
    task_bank.add_memory(mock_memory, mock_memory["task_description"])
    for st in mock_memory["subtasks"]:
        subtask_bank.add_memory(st, st["description"])
        
    task_bank.save(task_bank_path)
    subtask_bank.save(subtask_bank_path)
    
    # 2. Run Evaluation
    eval_pipeline = EvaluationPipeline(task_bank_path, subtask_bank_path)
    
    test_tasks = [
        {
            "description": (
                "Organize a team lunch for next Friday at 12:30 PM. "
                "Check Bob and Charlie's calendars first."
            )
        }
    ]
    
    config = {
        "model": "gpt-4o",
        "K": 5,
        "temperature": 0,
        "retrieval_strategy": "Vanilla"
    }
    
    success_rate = eval_pipeline.run_eval(test_tasks, config)
    print(f"Reproduction Success Rate: {success_rate * 100}%")

if __name__ == "__main__":
    reproduce()
