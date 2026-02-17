import os
from legomem.memory.vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

def seed_bench_memories():
    print("Seeding Benchmark Memories...")
    os.makedirs("data/memory_bank", exist_ok=True)
    
    task_bank = VectorStore()
    subtask_bank = VectorStore()
    
    memories = [
        {
            "task_description": "What is Bob's specific internal ID mentioned in the LEGOMem Standard Protocol?",
            "high_level_plan": "1. Access LEGOMem Personnel Records. 2. Retrieve Bob's ID 'B-99'.",
            "subtasks": [
                {"agent": "id_agent", "description": "Retrieve Bob's ID 'B-99'"}
            ]
        }
    ]
    
    for m in memories:
        task_bank.add_memory(m, m["task_description"])
        for st in m["subtasks"]:
            subtask_bank.add_memory(st, st["description"])
            
    task_bank.save("data/memory_bank/task_bank")
    subtask_bank.save("data/memory_bank/subtask_bank")
    print("Memories seeded successfully.")

if __name__ == "__main__":
    seed_bench_memories()
