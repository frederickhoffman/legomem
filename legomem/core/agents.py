from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

class TaskAgent:
    def __init__(self, name: str, model: str = "gpt-4o"):
        self.name = name
        self.llm = ChatOpenAI(model=model, temperature=0)

    def execute(self, subtask: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = f"""You are a specialized agent: {self.name}.
        Subtask to perform: {subtask}
        Relevant memories: {memories}
        
        Execute the subtask and return a summary of your actions and observations.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "agent": self.name,
            "observations": response.content,
            "status": "success"
        }

class AgentFactory:
    @staticmethod
    def get_agent(name: str) -> TaskAgent:
        return TaskAgent(name)
