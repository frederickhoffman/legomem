import os

from dotenv import load_dotenv

from legomem.eval.evaluator import EvaluationPipeline
from legomem.memory.vector_store import VectorStore

load_dotenv()

def reproduce():
    print("Starting LEGOMem Reproduction...")
    
    # 1. Memories are assumed to be seeded by seed_bench.py
    task_bank_path = "data/memory_bank/task_bank"
    subtask_bank_path = "data/memory_bank/subtask_bank"
    print(f"Loading memories from {task_bank_path}...")
    
    # 2. Run Evaluation (LEGOMem)
    print("\n--- Evaluating WITH LEGOMem ---")
    eval_pipeline = EvaluationPipeline(task_bank_path, subtask_bank_path)
    
    test_tasks = [
        {
            "description": "What is Bob's specific internal ID mentioned in the LEGOMem Standard Protocol?",
            "expected_output": "The internal ID for Bob is B-99."
        }
    ]
    
    config = {
        "model": "gpt-4o",
        "K": 5,
        "temperature": 0,
        "retrieval_strategy": "Vanilla",
        "mode": "LEGOMem"
    }
    
    success_rate_lego = eval_pipeline.run_eval(test_tasks, config)
    print(f"LEGOMem Success Rate: {success_rate_lego * 100}%")

    # 3. Run Evaluation (Baseline - No Memory)
    print("\n--- Evaluating WITHOUT Memory (Baseline) ---")
    # Empty paths to simulate no memory
    eval_pipeline_baseline = EvaluationPipeline("data/empty", "data/empty")
    os.makedirs("data/empty", exist_ok=True)
    
    config_baseline = {
        "model": "gpt-4o",
        "K": 0,
        "temperature": 0,
        "retrieval_strategy": "Vanilla",
        "mode": "Baseline"
    }
    
    success_rate_baseline = eval_pipeline_baseline.run_eval(test_tasks, config_baseline)
    print(f"Baseline Success Rate: {success_rate_baseline * 100}%")

    print("\n--- Summary ---")
    print(f"Baseline: {success_rate_baseline * 100}%")
    print(f"LEGOMem: {success_rate_lego * 100}%")
    print(f"Improvement: {(success_rate_lego - success_rate_baseline) * 100}%")

if __name__ == "__main__":
    reproduce()
