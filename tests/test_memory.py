import tempfile
import unittest
from pathlib import Path

from enterprise_rag_agent.config import Settings
from enterprise_rag_agent.llm import MockLLM
from enterprise_rag_agent.memory import MemoryManager, SessionStore
from enterprise_rag_agent.models import ChatMessage


class MemoryTest(unittest.TestCase):
    def test_summary_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions.json")
            settings = Settings(max_history_messages=2, summary_enabled=True, context_max_tokens=1000)
            manager = MemoryManager(store, settings, MockLLM())
            for i in range(4):
                store.add_message(store.get_or_create("s").session_id, ChatMessage(role="user", content=f"问题{i}"))
            context = manager.prepare("s", "问题4")
            self.assertTrue(context.summary)
            self.assertLessEqual(len(context.messages), 4)
            found = store.search("s", "问题2")
            self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()

