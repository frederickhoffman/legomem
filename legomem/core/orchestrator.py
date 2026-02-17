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
    final_answer: str | None

class Orchestrator:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def plan(self, state: AgentState) -> dict[str, Any]:
        """Generate or refine a high-level plan based on memories."""
        memory_context = "\n".join([str(m) for m in state.get("memories", [])])
        prompt = f"""You are an orchestrator agent. Your task is to solve: {state['task_description']}
        
        Successful trajectories from similar past tasks:
        {memory_context}
        
        Based on these experiences, generate a high-level plan as a numbered list of subtasks.
        Focus only on the plan, one subtask per line.
        """
        
        if not state.get("plan"):
             response = self.llm.invoke([HumanMessage(content=prompt)])
             plan = [s.strip() for s in response.content.split("\n") if s.strip() and s[0].isdigit()]
             return {"plan": plan, "current_step": 0}
        return {}

    def delegate(self, state: AgentState) -> dict[str, Any]:
        """Delegate the current subtask to a task agent."""
        if state["current_step"] >= len(state["plan"]):
            return {"final_answer": "Task completed."}
        
        subtask = state["plan"][state["current_step"]]
        
        # In the LEGOMem framework, we'd select a specialized agent here.
        # For simplicity in this implementation, we use a generic executor 
        # that follows the subtask description.
        prompt = f"""Execute this subtask: {subtask}
        Context of the main task: {state['task_description']}
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "messages": [HumanMessage(content=f"Subtask Outcome: {response.content}")],
            "current_step": state["current_step"] + 1
        }

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
