# Research Assistant

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
```

启动 Web 服务：

```powershell
python web_app.py 8000
```

开发环境默认访问：

```text
http://127.0.0.1:8000
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
tests/                             文献分析和 Web 层回归测试
```

## 测试

```powershell
python -m pytest -q
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
