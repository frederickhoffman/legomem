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
    def __init__(self, model: str = "gpt-4o", worker_model: str = None):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.worker_llm = ChatOpenAI(model=worker_model or model, temperature=0)

    def plan(self, state: AgentState) -> dict[str, Any]:
        """Generate or refine a high-level plan based on memories."""
        memory_list = state.get("memories", [])
        memory_context = ""
        for i, m in enumerate(memory_list):
            m_text = f"Memory {i+1}:\nTask: {m.get('task_description')}\nPlan: {m.get('high_level_plan')}"
            memory_context += m_text + "\n\n"
            
        prompt = (
            "You are an orchestrator agent. "
            f"Solve the task: {state['task_description']}\n\n"
            "MANDATORY REFERENCE (follow any specific codes/IDs/protocols here):"
            f"\n{memory_context}\n\n"
            "Task: {state['task_description']}\n"
            "Based on the referencing memories, generate a high-level plan. "
            "CRITICAL: You MUST COPY specific values (names, IDs, codes, years) from the memories into your plan steps. "
            "Do not be generic. usage: 'Verify ID B-99' instead of 'Verify ID'. "
            "Respond only with a numbered list of subtasks."
        )
        
        if not state.get("plan"):
             response = self.llm.invoke([HumanMessage(content=prompt)])
             plan = [
                 s.strip() for s in response.content.split("\n") 
                 if s.strip() and s[0].isdigit()
             ]
             return {"plan": plan, "current_step": 0}
        return {}

    def delegate(self, state: AgentState) -> dict[str, Any]:
        """Delegate the current subtask to a task agent."""
        subtask = state["plan"][state["current_step"]]
        prompt = f"Execute this subtask: {subtask}\nContext: {state['task_description']}"
        response = self.worker_llm.invoke([HumanMessage(content=prompt)])
        return {
            "messages": [HumanMessage(content=f"Subtask Outcome: {response.content}")],
            "current_step": state["current_step"] + 1
        }

    def summarize(self, state: AgentState) -> dict[str, Any]:
        """Provide a final summary of the completed task."""
        messages = state.get("messages", [])
        history = "\n".join([m.content for m in messages if isinstance(m, HumanMessage)])
        final_prompt = (
            f"Task: {state['task_description']}\n"
            f"Work History:\n{history}\n\n"
            "Provide the final success confirmation. Be specific about any IDs, "
            "protocols, or outcomes achieved."
        )
        response = self.llm.invoke([HumanMessage(content=final_prompt)])
        return {"final_answer": response.content}

def create_legomem_graph(orchestrator: Orchestrator):
    def should_continue(state: AgentState):
        if state.get("final_answer"):
            return END
        if state["current_step"] >= len(state.get("plan", [])):
            return "summarizer"
        return "delegator"

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", orchestrator.plan)
    workflow.add_node("delegator", orchestrator.delegate)
    workflow.add_node("summarizer", orchestrator.summarize)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", should_continue)
    workflow.add_conditional_edges("delegator", should_continue)
    return workflow.compile()
