import os
from legomem.memory.vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

def seed_bench_memories():
    print("Seeding Scaled Benchmark Memories...")
    os.makedirs("data/memory_bank", exist_ok=True)
    
    task_bank = VectorStore()
    subtask_bank = VectorStore()
    
    memories = [
        {
            "task_description": "What is Bob's specific internal ID mentioned in the LEGOMem Standard Protocol?",
            "high_level_plan": "1. Access LEGOMem Personnel Records. 2. Retrieve Bob's ID 'B-99'.",
            "subtasks": [{"agent": "id_agent", "description": "Retrieve Bob's ID 'B-99'"}]
        },
        {
            "task_description": "What is the encrypted activation code for Protocol-X in Bob's calendar records?",
            "high_level_plan": "1. Access Calendar Protocols. 2. Locate Protocol-X entry. 3. Extract code 'X-777'.",
            "subtasks": [{"agent": "cal_agent", "description": "Extract Protocol-X code 'X-777'"}]
        },
        {
            "task_description": "Identify the primary expense auditor for the Q4 Travel Budget as per LEGOMem Finance Policy.",
            "high_level_plan": "1. Open Finance Policy doc. 2. Scroll to Q4 Travel section. 3. Identify Sarah Jenkins as auditor.",
            "subtasks": [{"agent": "fin_agent", "description": "Identify Sarah Jenkins as Q4 auditor"}]
        },
        {
            "task_description": "What is the mandatory retention period for Project Icarus documentation according to the Compliance Protocol?",
            "high_level_plan": "1. Load Compliance Protocol. 2. Search for Project Icarus. 3. Confirm 7-year retention period.",
            "subtasks": [{"agent": "comp_agent", "description": "Confirm 7-year retention for Icarus"}]
        },
        {
            "task_description": "Who is the designated emergency contact for Server Room B in the LEGOMem Security Manual?",
            "high_level_plan": "1. Open Security Manual. 2. Locate 'Server Room B' chapter. 3. Identify Mike Miller as emergency contact.",
            "subtasks": [{"agent": "sec_agent", "description": "Identify Mike Miller for Server Room B"}]
        }
    ]
    
    for m in memories:
        task_bank.add_memory(m, m["task_description"])
        for st in m["subtasks"]:
            subtask_bank.add_memory(st, st["description"])
            
    task_bank.save("data/memory_bank/task_bank")
    subtask_bank.save("data/memory_bank/subtask_bank")
    print("Multi-task memories seeded successfully.")

if __name__ == "__main__":
    seed_bench_memories()
