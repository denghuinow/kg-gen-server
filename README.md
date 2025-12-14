# kg-gen-server

独立的知识图谱构建与预览服务（FastAPI），用于将 Markdown 文本通过 ATOM 抽取为知识图谱并写入 Neo4j，同时提供图谱列表/统计/可视化/删除/重命名等接口。

## 目录结构

```
kg-gen-server/
├── server.py
├── config.yaml
├── templates/
│   └── build_test.html
├── requirements.txt
└── README.md
```

## 安装与启动（uv）

1) 进入目录并创建虚拟环境：

```bash
cd kg-gen-server
uv venv
```

2) 安装依赖（包含对仓库内 `itext2kg` 的可编辑引用）：

```bash
uv pip install -r requirements.txt
```

3) 配置文件：

- 默认读取 `kg-gen-server/config.yaml`
- 也可通过环境变量指定：`KG_GEN_SERVER_CONFIG=/path/to/config.yaml`

4) 启动服务：

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8008 --reload
```

## 外部依赖

- `chonkie-fastapi`：需要在 `config.yaml` 的 `text.chunker.url` 配置可访问的 `SentenceChunker` 接口。
- LLM/Embeddings：从 `config.yaml` 的 `llm` / `embeddings` 读取（建议使用 `*_env` 从环境变量注入 key）。
- Neo4j：从 `config.yaml` 的 `neo4j` 读取连接信息。

## API

- `POST /api/graph/build`：构建/合并图谱并写入 Neo4j（支持溯源字段 `doc_id` 数组）
- `GET /api/graphs`：列出图谱（含节点/关系统计）
- `GET /api/graphs/{graph_name}/stats`：指定图谱统计
- `GET /api/graphs/{graph_name}/visualize`：指定图谱可视化 HTML（可在 iframe 中使用）
- `DELETE /api/graphs/{graph_name}`：删除图谱
- `POST /api/graphs/{graph_name}/rename`：重命名图谱
- `GET /` 或 `GET /test`：测试页面（表单 + 预览 iframe）

## 溯源说明（doc_id）

- 节点与关系都使用 `doc_id` 数组属性存储来源文档 ID。
- 写入时使用 `COALESCE(..., [])` 做数组合并与去重（仅对本次构建“触达”的关系及其两端实体更新 doc_id，避免误把旧图谱内容归因到新文档）。
