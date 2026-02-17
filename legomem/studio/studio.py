import os

from legomem.core.orchestrator import Orchestrator, create_legomem_graph
from legomem.memory.vector_store import VectorStore


# Configuration for Studio
def get_graph(model: str = "gpt-4o", k: int = 5):
    # Load default memory banks if they exist
    task_bank = VectorStore()
    if os.path.exists("data/memory_bank/task_bank.index"):
        task_bank.load("data/memory_bank/task_bank")
        
    subtask_bank = VectorStore()
    if os.path.exists("data/memory_bank/subtask_bank.index"):
        subtask_bank.load("data/memory_bank/subtask_bank")
        
    orchestrator = Orchestrator(model=model)
    return create_legomem_graph(orchestrator)

# This is the entry point for LangGraph Studio
graph = get_graph()
