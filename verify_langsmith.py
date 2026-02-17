"""Verify LangSmith connectivity and tracing."""
import os
from dotenv import load_dotenv

load_dotenv()


def verify_langsmith():
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("❌ LANGCHAIN_API_KEY not found in .env.")
        return

    # 1. Verify client connectivity
    from langsmith import Client
    client = Client()
    print(f"✅ LangSmith Client initialized.")
    print(f"   Endpoint: {client.api_url}")
    print(f"   Project:  {os.getenv('LANGCHAIN_PROJECT', 'default')}")

    # 2. Run a minimal traced LangGraph invocation
    print("\n--- Running traced LangGraph invocation ---")
    from legomem.core.orchestrator import Orchestrator, create_legomem_graph

    orchestrator = Orchestrator(model="gpt-4o")
    app = create_legomem_graph(orchestrator)

    result = app.invoke({
        "task_description": "What is 2 + 2?",
        "memories": [],
        "messages": [],
        "plan": [],
        "current_step": 0,
        "final_answer": None,
    })

    print(f"   Agent answer: {result.get('final_answer', 'N/A')[:80]}...")
    print("✅ Traced run completed. Check your LangSmith dashboard for the trace.")


if __name__ == "__main__":
    verify_langsmith()
