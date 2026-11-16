#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Run a VSAG-backed LangChain retrieval workflow without network access.

Install the optional dependencies first:

    pip install "pyvsag[langchain]"

Then run:

    python examples/python/111_langchain_vector_store.py
"""

import hashlib

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pyvsag.langchain import VSAGVectorStore


class HashingEmbedding(Embeddings):
    """Deterministic hashed bag-of-words embedding: local, fast and credential free."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            token = "".join(char for char in token if char.isalnum())
            if not token:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def main() -> None:
    documents = [
        Document(
            page_content="The Eiffel Tower is a landmark in Paris.",
            metadata={"source": "landmarks/eiffel_tower.md", "topic": "landmark"},
        ),
        Document(
            page_content="The Great Wall is a landmark in China.",
            metadata={"source": "landmarks/great_wall.md", "topic": "landmark"},
        ),
        Document(
            page_content="Photosynthesis converts light into energy in plants.",
            metadata={"source": "science/photosynthesis.md", "topic": "science"},
        ),
        Document(
            page_content="Gravity keeps the planets in orbit around the Sun.",
            metadata={"source": "science/gravity.md", "topic": "science"},
        ),
    ]

    store = VSAGVectorStore.from_documents(documents, HashingEmbedding())

    print("== similarity_search ==")
    for document in store.similarity_search("which landmark is in Paris", k=2):
        print(f"- {document.page_content}")
        print(f"  metadata={document.metadata} id={document.id}")

    print("\n== as_retriever ==")
    retriever = store.as_retriever(search_kwargs={"k": 2})
    for document in retriever.invoke("how do plants convert light into energy"):
        print(f"- {document.page_content}")
        print(f"  metadata={document.metadata}")


if __name__ == "__main__":
    main()
