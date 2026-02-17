from typing import Any

from ..core.orchestrator import Orchestrator, create_legomem_graph
from ..memory.retrieval import MemoryRetriever
from ..memory.vector_store import VectorStore
from ..monitoring.wandb_logger import WandBLogger


class EvaluationPipeline:
    def __init__(self, task_bank_path: str, subtask_bank_path: str):
        self.task_bank = VectorStore()
        self.task_bank.load(task_bank_path)
        
        self.subtask_bank = VectorStore()
        self.subtask_bank.load(subtask_bank_path)
        
        self.retriever = MemoryRetriever(self.task_bank, self.subtask_bank)
        self.logger = WandBLogger()

    def run_eval(self, tasks: list[dict[str, Any]], config: dict[str, Any]):
        self.logger.start_run(config)
        
        success_count = 0
        for task in tasks:
            print(f"Running task: {task['description']}")
            
            # Retrieval
            memories = self.retriever.retrieve_vanilla(task['description'])
            
            # Setup State
            orchestrator = Orchestrator(model=config.get("model", "gpt-4o"))
            app = create_legomem_graph(orchestrator)
            
            inputs = {
                "task_description": task['description'],
                "memories": memories,
                "messages": [],
                "plan": [],
                "current_step": 0
            }
            
            # Run Agent
            app.invoke(inputs)
            
            # Mock Success check
            is_success = True # In reality, check result against task['expected_output']
            if is_success:
                success_count += 1
            
            self.logger.log_metrics({"task_success": int(is_success)})

        success_rate = success_count / len(tasks) if tasks else 0
        self.logger.log_metrics({"total_success_rate": success_rate})
        self.logger.finish_run()
        
        return success_rate
