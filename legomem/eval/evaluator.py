from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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

    def _verify_success(self, task: dict[str, Any], result: dict[str, Any], model: str) -> bool:
        """Verify task success using an LLM judge."""
        llm = ChatOpenAI(model=model, temperature=0)
        
        # Safe access to result outcomes
        actual_output = result.get('final_answer')
        if not actual_output and result.get('messages'):
            actual_output = result['messages'][-1].content
        if not actual_output:
            actual_output = "No outcome"

        prompt = (
            "You are a rigorous evaluation judge. "
            "Task: {description}\n"
            "Expected Outcome: {expected}\n"
            "Agent's Final Answer/Outcome: {actual}\n\n"
            "Final Answer: Does the agent's outcome match the expected outcome? "
            "Respond with only 'YES' or 'NO'."
        ).format(
            description=task['description'],
            expected=task.get('expected_output', 'Success'),
            actual=actual_output
        )
        
        # The 'state' variable is not available in _verify_success.
        # Memories are part of the input to the orchestrator, not its output 'result'.
        # To print memories here, they would need to be passed as an argument to this method.
        # For now, we will only add the judge prompt and response debug prints as requested.
        
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"DEBUG: Judge Prompt: {prompt}")
        print(f"DEBUG: Judge Response: {response.content}")
        return "YES" in response.content.upper()

    def run_eval(self, tasks: list[dict[str, Any]], config: dict[str, Any]):
        self.logger.start_run(config)
        
        success_count = 0
        model = config.get("model", "gpt-4o")
        
    def run_single_task(self, task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Run a single task through the LEGOMem system."""
        model = config.get("model", "gpt-4o")
        
        # Retrieval logic
        strategy = config.get("retrieval_strategy", "Vanilla")
        k = config.get("K", 5)
        
        if strategy == "Vanilla":
            memories = self.retriever.retrieve_vanilla(task['description'], k=k)
        else:
            memories = self.retriever.retrieve_vanilla(task['description'], k=k)
            
        # Setup State
        orchestrator = Orchestrator(model=model)
        app = create_legomem_graph(orchestrator)
        
        inputs = {
            "task_description": task['description'],
            "memories": memories,
            "messages": [],
            "plan": [],
            "current_step": 0,
            "final_answer": None
        }
        
        # Run Agent
        return app.invoke(inputs)

    def run_eval(self, tasks: list[dict[str, Any]], config: dict[str, Any]):
        self.logger.start_run(config)
        
        success_count = 0
        model = config.get("model", "gpt-4o")
        
        for task in tasks:
            print(f"Running task: {task['description']}")
            result = self.run_single_task(task, config)
            
            # Rigorous Success check
            is_success = self._verify_success(task, result, model)
            if is_success:
                success_count += 1
            
            print(f"Task Success: {is_success}")
            self.logger.log_metrics({"task_success": int(is_success)})

        success_rate = success_count / len(tasks) if tasks else 0
        self.logger.log_metrics({"total_success_rate": success_rate})
        self.logger.finish_run()
        
        return success_rate
