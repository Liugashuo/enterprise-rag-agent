# 基于 RAG 的企业文档智能 Agent

一个可运行的企业文档智能问答系统，技术栈为 Python、LLM、RAG、LangGraph、Embedding、向量数据库、Reranker、FastAPI。

## 能力概览

- **LangGraph 工作流**：意图识别 → 任务规划 → 工具调用 → 证据校验 → 答案生成。
- **RAG Pipeline**：支持 PDF、Word、Markdown/TXT 解析、递归分块、Embedding、向量入库和增量更新。
- **Hybrid Search + Rerank**：BM25 + 向量检索，RRF 融合；支持关键词重排与 CrossEncoder 重排。
- **Query Rewrite**：多轮对话指代消解与上下文补全，可切换 LLM 重写或规则重写。
- **Tool Calling**：知识库检索、历史消息查询封装为 Agent Tools，由 LLM 动态选择调用。
- **短期记忆**：多轮会话、历史摘要、Token 预算内的上下文压缩。
- **可信度与约束**：证据校验、低置信度提示、最大 Agent 步数、上下文截断，降低幻觉与推理成本。

## 目录结构

```text
src/enterprise_rag_agent/
├── api.py                 # FastAPI 接口
├── config.py              # 配置加载（环境变量 + YAML）
├── llm.py                 # DeepSeek / OpenAI 兼容 LLM 与 Mock 演示模型
├── service.py             # 服务编排：入库、问答、文档管理
├── document/              # 文档解析、分块、文档目录
├── vectorstore/           # JsonVectorStore / ChromaVectorStore
├── retrieval/             # BM25、Hybrid、Rerank、Query Rewrite
├── memory/                # 会话存储、摘要、上下文管理
├── tools/                 # 知识检索与历史检索工具
└── graph/                 # LangGraph 状态、节点、工作流
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[all]"
python run.py
```

默认配置优先使用 DeepSeek，但未配置 `DEEPSEEK_API_KEY` 时会自动降级到 `MockLLM`，
同时使用 `hash` 本地嵌入和 `json` 向量库，因此无需 API Key 也能完整跑通演示。

接入真实 LLM / Embedding：

```bash
copy .env.example .env
# 编辑 .env，设置 DEEPSEEK_API_KEY
# DeepSeek 默认模型为 deepseek-chat，接口为 https://api.deepseek.com/v1
# Embedding 如需语义向量，可配置其他 OpenAI 兼容 Embedding 服务
```

## 文档入库

```python
from enterprise_rag_agent.service import RAGAgentService

service = RAGAgentService()
print(service.ingest_file("examples/company_policy.md"))
print(service.chat("公司年假有多少天？"))
```

也可以直接运行演示：

```bash
python examples/demo.py
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/v1/chat` | 多轮问答，`{"session_id": "可选", "message": "问题"}` |
| POST | `/v1/documents` | 上传文件入库（multipart） |
| POST | `/v1/documents/ingest` | 按本地路径入库，`{"path": "..."}` |
| GET | `/v1/documents` | 文档列表 |
| DELETE | `/v1/documents/{document_id}` | 删除文档及向量 |
| GET | `/v1/sessions/{session_id}/messages` | 会话历史 |
| DELETE | `/v1/sessions/{session_id}` | 删除会话 |

示例：

```bash
curl -X POST http://localhost:8000/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"年假如何申请？\"}"
```

## 检索策略

1. `Query Rewrite` 将多轮问题补全为独立检索问题。
2. 向量检索返回语义相似片段，BM25 返回关键词命中片段。
3. `Reciprocal Rank Fusion` 融合两路结果。
4. 可选用 `BAAI/bge-reranker-base` 或关键词重合度进行重排。
5. Agent 从工具结果中选取证据，证据不足时明确提示低置信度。

## 生产建议

- 将 `VECTOR_STORE_TYPE` 切换为 `chroma`，并使用真实 Embedding。
- 将 `MODEL_NAME` 切换为具备 Function Calling 能力的模型。
- 将 `RERANKER_MODEL` 设置为本地 CrossEncoder 模型。
- 为上传接口增加鉴权、文件大小限制、病毒扫描和对象存储。
- 使用 Redis/PostgreSQL 等外部存储替换 JSON 会话与目录。
