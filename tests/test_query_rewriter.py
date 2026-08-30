import unittest

from enterprise_rag_agent.models import ChatMessage
from enterprise_rag_agent.retrieval.query_rewriter import RuleBasedQueryRewriter


class QueryRewriterTest(unittest.TestCase):
    def test_reference_resolution(self):
        rewriter = RuleBasedQueryRewriter()
        history = [ChatMessage(role="user", content="公司的年假制度是什么？")]
        query = "那它的申请流程呢？"
        result = rewriter.rewrite(query, history)
        self.assertIn("上文背景", result)


if __name__ == "__main__":
    unittest.main()

