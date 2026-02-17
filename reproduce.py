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
    
    # Load tasks from datasets
    from legomem.bench.datasets import OfficeBenchLoader
    loader = OfficeBenchLoader()
    test_tasks = loader.load_level(1)
    
    print(f"Loaded {len(test_tasks)} tasks for evaluation.")
    
    config = {
        "model": "gpt-4o",
        "K": 5,
        "temperature": 0,
        "retrieval_strategy": "Vanilla",
        "mode": "LEGOMem"
    }
    
    # Run evaluation manually to track split metrics
    print(f"Running LEGOMem evaluation on {len(test_tasks)} tasks...")
    lego_results = []
    for task in test_tasks:
        result = eval_pipeline.run_single_task(task, config)
        is_success = eval_pipeline._verify_success(task, result, config["model"])
        lego_results.append({
            "id": task["id"],
            "type": task.get("type", "unknown"),
            "success": is_success
        })
        print(f"Task {task['id']} ({task.get('type')}): {'SUCCESS' if is_success else 'FAILURE'}")

    # 3. Run Evaluation (Baseline - No Memory)
    print("\n--- Evaluating WITHOUT Memory (Baseline) ---")
    eval_pipeline_baseline = EvaluationPipeline("data/empty", "data/empty")
    os.makedirs("data/empty", exist_ok=True)
    
    config_baseline = {
        "model": "gpt-4o",
        "K": 0,
        "temperature": 0,
        "retrieval_strategy": "Vanilla",
        "mode": "Baseline"
    }
    
    baseline_results = []
    for task in test_tasks:
        result = eval_pipeline_baseline.run_single_task(task, config_baseline)
        is_success = eval_pipeline_baseline._verify_success(task, result, config_baseline["model"])
        baseline_results.append({
            "id": task["id"],
            "type": task.get("type", "unknown"),
            "success": is_success
        })
        print(f"Task {task['id']} ({task.get('type')}): {'SUCCESS' if is_success else 'FAILURE'}")

    # Calculate Metrics
    def calc_metrics(results):
        total = len(results)
        procedural = [r for r in results if r["type"] == "procedural"]
        general = [r for r in results if r["type"] == "general"]
        
        return {
            "total": sum(r["success"] for r in results) / total * 100 if total else 0,
            "procedural": sum(r["success"] for r in procedural) / len(procedural) * 100 if procedural else 0,
            "general": sum(r["success"] for r in general) / len(general) * 100 if general else 0
        }

    lego_metrics = calc_metrics(lego_results)
    baseline_metrics = calc_metrics(baseline_results)

    print("\n--- Final Reproduction Results ---")
    print(f"{'Metric':<20} | {'Baseline':<10} | {'LEGOMem':<10} | {'Delta':<10}")
    print("-" * 60)
    print(f"{'Overall Success':<20} | {baseline_metrics['total']:<10.1f}% | {lego_metrics['total']:<10.1f}% | {lego_metrics['total'] - baseline_metrics['total']:<+10.1f}%")
    print(f"{'Procedural (Mem)':<20} | {baseline_metrics['procedural']:<10.1f}% | {lego_metrics['procedural']:<10.1f}% | {lego_metrics['procedural'] - baseline_metrics['procedural']:<+10.1f}%")
    print(f"{'General (Reasoning)':<20} | {baseline_metrics['general']:<10.1f}% | {lego_metrics['general']:<10.1f}% | {lego_metrics['general'] - baseline_metrics['general']:<+10.1f}%")

    # 4. Run Evaluation (Hybrid Team - GPT-4o Planner + SLM Worker + QueryRewrite)
    print("\n--- Evaluating Hybrid Team (GPT-4o + GPT-4o-mini + QueryRewrite) ---")
    
    config_hybrid = {
        "model": "gpt-4o",
        "worker_model": "gpt-4o-mini",
        "K": 5,
        "temperature": 0,
        "retrieval_strategy": "QueryRewrite",
        "mode": "Hybrid"
    }
    
    hybrid_results = []
    for task in test_tasks:
        try:
            result = eval_pipeline.run_single_task(task, config_hybrid)
            is_success = eval_pipeline._verify_success(task, result, config_hybrid["model"])
        except Exception as e:
            print(f"Error running hybrid task {task['id']}: {e}")
            is_success = False
            
        hybrid_results.append({
            "id": task["id"],
            "type": task.get("type", "unknown"),
            "success": is_success
        })
        print(f"Task {task['id']} ({task.get('type')}): {'SUCCESS' if is_success else 'FAILURE'}")

    hybrid_metrics = calc_metrics(hybrid_results)
    
    print("\n--- Final Hybrid Team Results ---")
    print(f"Overall Success: {hybrid_metrics['total']:.1f}%")
    print(f"Procedural (Mem): {hybrid_metrics['procedural']:.1f}%")
    print(f"General (Reasoning): {hybrid_metrics['general']:.1f}%")

if __name__ == "__main__":
    reproduce()
