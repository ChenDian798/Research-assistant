# UI Handoff Notes

This package is prepared for UI design and frontend upgrade work.

## How to Run

```powershell
pip install -r requirements.txt
python web_app.py 8000
```

Development default:

```text
http://127.0.0.1:8000
```

For server deployment, use the configured domain, reverse-proxy URL, or server IP instead.

`start_web.bat` starts a development instance of the web app.

The package intentionally includes `.env` so the handoff can run in development or be adapted for server deployment.

## UI Entry Points

- `web/index.html`: page structure and forms.
- `web/styles.css`: all current styling.
- `web/app.js`: client-side state, form submission, rendering, polling, export controls.
- `web_app.py`: HTTP server, static file server, and API handlers.

## Main Pages

- Literature analysis: `#literature`

## Key APIs Used by the UI

- `POST /api/literature-analysis`
- `POST /api/literature-analysis/pdf`
- `GET /api/literature-analysis/{job_id}`
- `POST /api/export/pdf`

Most long-running operations return a `job_id`; the frontend polls the corresponding `GET` endpoint.

The backend exposes only literature analysis and PDF export endpoints.

## Important Current Behavior

Users may upload files that are not papers. For example, a DOCX may be a teacher's writing requirements, rubric, assignment prompt, style guide, or grading criteria.

Recent workflow logic classifies uploaded PDF/DOCX files before analysis:

- Literature-like files are treated as papers/references.
- Requirement/rubric/style files are passed as auxiliary context and constraints.
- Non-paper files should not be shown as analyzed papers in the literature table.

Relevant backend functions:

- `web_app.py`: `_infer_uploaded_document_role`
- `web_app.py`: `_split_reference_roles`
- `web_app.py`: `_build_uploaded_context`
- `src/research_agent/literature_workflow.py`: returns an empty table with a context-only summary when no real literature references remain

## Design Upgrade Notes

The current frontend is plain HTML/CSS/JS with no build step. A designer/frontend engineer can either:

- keep the no-build structure and redesign `web/index.html`, `web/styles.css`, and `web/app.js`; or
- introduce a framework/build system, while keeping `web_app.py` API contracts stable.

Useful sample artifacts are included under `sample_outputs/` in the handoff package so UI states can be designed without always running full LLM jobs.
