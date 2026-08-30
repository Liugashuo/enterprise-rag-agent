import unittest

from enterprise_rag_agent.models import Chunk
from enterprise_rag_agent.vectorstore import JsonVectorStore


class JsonVectorStoreTest(unittest.TestCase):
    def test_add_search_delete(self):
        store = JsonVectorStore("data/test_vector_store")
        store.clear()
        store.add(
            [Chunk(id="a", document_id="d1", content="hello world", metadata={})],
            [[1.0, 0.0]],
        )
        hits = store.search([1.0, 0.0], top_k=5)
        self.assertEqual(hits[0].chunk_id, "a")
        self.assertEqual(store.delete("d1"), 1)
        self.assertEqual(store.search([1.0, 0.0]), [])
        store.clear()


if __name__ == "__main__":
    unittest.main()

