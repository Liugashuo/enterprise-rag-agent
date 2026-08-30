import hashlib
import unittest

from enterprise_rag_agent.document import RecursiveTextChunker
from enterprise_rag_agent.document.loaders import ParsedDocument


class ChunkerTest(unittest.TestCase):
    def test_split_and_metadata(self):
        text = "第一段。" * 200 + "\n\n" + "第二段。" * 200
        chunker = RecursiveTextChunker(chunk_size=120, chunk_overlap=20)
        doc_id = hashlib.sha256(b"demo").hexdigest()
        chunks = chunker.split_document(ParsedDocument(text=text, filename="demo.md"), doc_id)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.document_id == doc_id for c in chunks))
        self.assertTrue(all(c.content for c in chunks))
        self.assertTrue(all(c.metadata["filename"] == "demo.md" for c in chunks))


if __name__ == "__main__":
    unittest.main()

