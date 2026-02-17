import argparse

from dotenv import load_dotenv

from legomem.bench.datasets import OfficeBenchLoader
from legomem.eval.evaluator import EvaluationPipeline

load_dotenv()

def run_benchmark(model: str = "gpt-4o", k: int = 5, strategies: list[str] = ["Vanilla"]):
    loader = OfficeBenchLoader()
    eval_pipeline = EvaluationPipeline(
        "data/memory_bank/task_bank",
        "data/memory_bank/subtask_bank"
    )
    
    datasets = loader.load_all_levels()
    
    for strategy in strategies:
        print(f"\n--- Running Benchmark with Strategy: {strategy} ---")
        for level, tasks in datasets.items():
            print(f"Evaluating {level} ({len(tasks)} tasks)...")
            config = {
                "model": model,
                "K": k,
                "retrieval_strategy": strategy,
                "dataset_level": level
            }
            success_rate = eval_pipeline.run_eval(tasks, config)
            print(f"{level} Success Rate: {success_rate * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LEGOMem Benchmarks")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategies", nargs="+", default=["Vanilla", "Dynamic"])
    
    args = parser.parse_args()
    run_benchmark(model=args.model, k=args.k, strategies=args.strategies)
