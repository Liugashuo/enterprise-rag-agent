"""企业文档智能 Agent 核心包。

为避免在没有安装 LangGraph / FastAPI 等可选运行时依赖时阻塞基础组件导入，
顶层不主动导入 service；需要时使用：
    from enterprise_rag_agent.service import RAGAgentService
    from enterprise_rag_agent.config import Settings
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
