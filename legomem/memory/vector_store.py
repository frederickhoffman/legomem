import os
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class VectorStore:
    def __init__(self, dimension: int = 3072): # 3072 for text-embedding-3-large
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index = faiss.IndexFlatL2(dimension)
        self.memories: list[dict[str, Any]] = []

    def _get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )
        return response.data[0].embedding

    def add_memory(self, content: dict[str, Any], text_to_embed: str):
        embedding = self._get_embedding(text_to_embed)
        vector = np.array([embedding]).astype("float32")
        self.index.add(vector)
        self.memories.append(content)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self._get_embedding(query)
        vector = np.array([query_embedding]).astype("float32")
        _, indices = self.index.search(vector, min(k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.memories[idx])
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        import json
        with open(f"{path}.json", "w") as f:
            json.dump(self.memories, f)

    def load(self, path: str):
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        if os.path.exists(f"{path}.json"):
            import json
            with open(f"{path}.json") as f:
                self.memories = json.load(f)
