from pathlib import Path

from enterprise_rag_agent.config import Settings
from enterprise_rag_agent.service import RAGAgentService


def main() -> None:
    settings = Settings.load()
    service = RAGAgentService(settings)
    sample = Path(__file__).with_name("company_policy.md")
    print("ingest:", service.ingest_file(sample))

    session_id = None
    for question in [
        "公司年假有多少天？",
        "那它的申请流程是什么？",
        "我刚才问了什么？",
    ]:
        result = service.chat(question, session_id=session_id)
        session_id = result["session_id"]
        print("\nQ:", question)
        print("A:", result["answer"])
        print("sources:", [s["document_id"] for s in result["sources"]])


if __name__ == "__main__":
    main()

