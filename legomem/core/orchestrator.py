from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

class AgentState(TypedDict):
    task_description: str
    plan: list[str]
    current_step: int
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    memories: list[dict[str, Any]]
    final_answer: Optional[str]

class Orchestrator:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def plan(self, state: AgentState) -> dict[str, Any]:
        """Generate or refine a high-level plan."""
        # In a real implementation, we'd use the memories here
        prompt = f"""You are an orchestrator agent. Your task is to solve: {state['task_description']}
        Retrieved memories: {state['memories']}
        
        Based on these memories and the task, generate a high-level plan.
        Return a list of subtasks.
        """
        # Simplification for now
        if not state.get("plan"):
             # Mock plan generation
             new_plan = ["Check calendar", "Send email"]
             return {"plan": new_plan, "current_step": 0}
        return {}

    def delegate(self, state: AgentState) -> dict[str, Any]:
        """Delegate the current subtask to a task agent."""
        if state["current_step"] >= len(state["plan"]):
            return {"final_answer": "Task completed."}
        
        subtask = state["plan"][state["current_step"]]
        # This would normally involve tool calling or agent invocation
        # For now, we simulate an agent response
        return {"messages": [HumanMessage(content=f"Subtask: {subtask}")], "current_step": state["current_step"] + 1}

def create_legomem_graph(orchestrator: Orchestrator):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", orchestrator.plan)
    workflow.add_node("delegator", orchestrator.delegate)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "delegator")
    
    def should_continue(state: AgentState):
        if state.get("final_answer") or state["current_step"] >= len(state["plan"]):
            return END
        return "delegator"

    workflow.add_conditional_edges("delegator", should_continue)
    
    return workflow.compile()
