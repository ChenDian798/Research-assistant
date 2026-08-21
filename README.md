# Research Assistant

Public deployment note: before exposing this app to the internet, apply the reverse-proxy template in `deploy/nginx-research-agent.conf`. It sets `client_max_body_size 30m`, matching `MAX_UPLOAD_TOTAL_MB=30`; see `deploy/README.md`.

当前项目只保留一个核心 Web 功能：**文献分析**。

文献分析支持输入 DOI、PMID、arXiv 链接、论文页面链接，或上传 PDF/DOCX，生成结构化文献分析表、跨文献总结和可导出的 Markdown/TXT/PDF 报告。

## 快速开始

安装运行依赖：

```powershell
pip install -r requirements.txt
```

如需运行测试，安装开发依赖：

```powershell
pip install -r requirements-dev.txt
```

创建配置文件：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，填入模型服务配置：

```text
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=your_model
OPENAI_BASE_URL=https://your-compatible-endpoint/v1
WEB_HOST=0.0.0.0
RESEARCH_AGENT_CONTACT_EMAIL=admin@example.com
ALLOW_SELF_REGISTRATION=true
SESSION_TTL_SECONDS=28800
```

启动 Web 服务：

```powershell
python web_app.py 8000
```

开发环境默认访问：

```text
http://127.0.0.1:8000
```

## 用户登录和管理员

当前 Web 应用内置轻量账号系统：普通用户可在页面内注册/登录，登录后的历史记录、任务和上传文件会按用户隔离保存。

创建第一个管理员账号：

```powershell
python scripts/create_admin.py admin@example.com
```

脚本会在终端安全提示输入密码。管理员登录后，顶部导航会显示“管理”，可查看用户列表、启用/禁用账号、调整管理员权限或删除用户数据。

管理员还可以进入“评估”页面批量运行检索题目。该页面支持一次输入多行题目、设置来源/模式/并发数，并会记录每条题目的总耗时、规划、召回、元数据修复、筛选、验证等阶段耗时，以及合格/待复核/过滤/原始候选数量。评估结果可导出 CSV，用于后续性能和检索质量优化。

可选配置：

```text
ALLOW_SELF_REGISTRATION=true
SESSION_TTL_SECONDS=28800
# Set false only for local HTTP testing. Keep true behind HTTPS in production.
SESSION_COOKIE_SECURE=true
```

## Web 模块

前端入口：

- `文献分析`：`literaturePage`

## API

主要接口：

- `POST /api/literature-analysis`
- `POST /api/literature-analysis/pdf`
- `GET /api/literature-analysis/{job_id}`
- `POST /api/export/pdf`

## Durable production runtime

Production uses PostgreSQL as the source of truth, Redis/Celery for execution,
and private S3/MinIO storage for uploaded documents and generated exports. Start
the backing services with `docker compose -f docker-compose.persistence.yml up -d`,
set the matching `DATABASE_URL`, `CELERY_BROKER_URL`, and S3 variables in `.env`,
then start a worker with:

```powershell
celery -A src.research_agent.web_tasks.celery_app worker --loglevel=INFO --queues=research-jobs
```

The API writes a queued job to the database before it submits the Celery message.
Workers re-load `request_json`, append `job_events`, and persist the terminal
result/error. `Idempotency-Key` is supported on long-running POST endpoints.
`history_records.json` is imported once into `history_entries` (a backup is kept)
and is not the authenticated production read path.

长任务会返回 `job_id`，前端通过对应的 `GET` 接口轮询状态。

可选环境变量：

```text
NCBI_EMAIL=you@example.com
NCBI_API_KEY=optional_ncbi_key
```

## 项目结构

```text
web_app.py                         Web API、任务轮询、静态文件服务、上传解析和 PDF 导出
web/                               无构建步骤的前端页面
src/research_agent/literature_workflow.py
                                   文献分析 workflow
src/research_agent/llm.py          OpenAI-compatible LLM 客户端
src/research_agent/citations.py    APA / IEEE / BibTeX 引用格式化
src/research_agent/doi.py          DOI、arXiv、PMID、网页元数据补全
src/research_agent/pubmed_search.py
                                   PMID 元数据获取
non_runtime/                       示例、测试题集、交接材料和回归测试
```

## 测试

```powershell
python -m pytest non_runtime/tests -q
```

当前测试覆盖：

- 文献分析上传、摘要片段提取和结构化结果归一化
- 引用格式化和 Markdown 表格解析
- PDF 导出辅助逻辑
- Web 层只保留文献分析、上传解析和 PDF 导出相关接口

## 已知技术债

- `web_app.py` 仍使用 `cgi.FieldStorage` 解析 multipart 上传；`cgi` 将在 Python 3.13 移除。
- Web 任务状态目前保存在进程内存 `JOBS` 字典中，服务重启会丢失任务，长时间运行也需要清理策略。
- `web_app.py`、`web/app.js` 仍偏大，后续可以按路由、任务、上传、导出、API client、渲染和状态管理拆分。
