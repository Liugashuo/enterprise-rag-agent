import unittest

from enterprise_rag_agent.embeddings import HashEmbedder
from enterprise_rag_agent.models import Chunk
from enterprise_rag_agent.retrieval import HybridRetriever, get_reranker
from enterprise_rag_agent.vectorstore import JsonVectorStore


class HybridTest(unittest.TestCase):
    def test_search_returns_evidence(self):
        store = JsonVectorStore("data/test_hybrid_store")
        store.clear()
        chunks = [
            Chunk(id="1", document_id="d1", content="公司规定员工每年享有十天年假。", metadata={}),
            Chunk(id="2", document_id="d1", content="新员工入职需要提交身份证复印件。", metadata={}),
        ]
        embedder = HashEmbedder(128)
        store.add(chunks, embedder.embed_documents([c.content for c in chunks]))
        retriever = HybridRetriever(
            vector_store=store,
            embedder=embedder,
            reranker=get_reranker(True, "keyword"),
            top_k=2,
        )
        results = retriever.search("年假有多少天？")
        self.assertTrue(results)
        self.assertIn("年假", results[0].content)
        store.clear()


if __name__ == "__main__":
    unittest.main()

