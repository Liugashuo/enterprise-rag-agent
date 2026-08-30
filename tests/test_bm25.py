import unittest

from enterprise_rag_agent.models import Chunk
from enterprise_rag_agent.retrieval.bm25 import BM25Index


class BM25Test(unittest.TestCase):
    def test_retrieves_relevant_chunk(self):
        chunks = [
            Chunk(id="1", document_id="d", content="员工年假制度为每年十天。", metadata={}),
            Chunk(id="2", document_id="d", content="服务器每天凌晨两点自动备份。", metadata={}),
            Chunk(id="3", document_id="d", content="报销发票需在一个月内提交。", metadata={}),
        ]
        index = BM25Index()
        index.build(chunks)
        results = index.search("年假有几天", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0][0], "1")


if __name__ == "__main__":
    unittest.main()

