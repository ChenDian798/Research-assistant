import { useEffect, useMemo, useState } from "react";
import { createSearchEvaluation, evaluationExportUrl, fetchEvaluationDetail, fetchEvaluations } from "../lib/api.js";

const sourceOptions = ["arxiv", "pubmed", "openalex", "crossref"];

export default function EvaluationView({ t }) {
  const [form, setForm] = useState({
    name: "",
    queryText: "",
    sources: ["arxiv", "pubmed"],
    searchMode: "auto",
    limit: "5",
    year: "",
    concurrency: "3",
    includeNeedsReview: true,
  });
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [detail, setDetail] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [isError, setIsError] = useState(false);

  const selectedRun = detail || runs.find((run) => run.id === selectedRunId) || null;
  const isActive = selectedRun && ["queued", "running"].includes(selectedRun.status);

  useEffect(() => {
    loadRuns();
  }, []);

  useEffect(() => {
    if (!selectedRunId) return undefined;
    loadDetail(selectedRunId, { silent: true });
    const timer = window.setInterval(() => loadDetail(selectedRunId, { silent: true }), isActive ? 1800 : 5000);
    return () => window.clearInterval(timer);
  }, [selectedRunId, isActive]);

  async function loadRuns() {
    setIsLoading(true);
    setIsError(false);
    try {
      const nextRuns = await fetchEvaluations();
      setRuns(nextRuns);
      const nextId = selectedRunId || nextRuns[0]?.id || "";
      setSelectedRunId(nextId);
      if (nextId) await loadDetail(nextId, { silent: true });
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      setIsLoading(false);
    }
  }

  async function loadDetail(runId, options = {}) {
    if (!options.silent) setIsLoading(true);
    try {
      const payload = await fetchEvaluationDetail(runId);
      setDetail(payload);
      setRuns((current) => [payload, ...current.filter((run) => run.id !== payload.id)]);
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      if (!options.silent) setIsLoading(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const sources = form.sources.join(",");
    setStatus(t("eval.starting"));
    setIsError(false);
    try {
      const payload = await createSearchEvaluation({
        name: form.name,
        query_text: form.queryText,
        sources,
        search_mode: form.searchMode,
        max_results_per_source: Number(form.limit || 5),
        year: form.year,
        concurrency: Number(form.concurrency || 3),
        include_needs_review: form.includeNeedsReview,
      });
      setSelectedRunId(payload.run_id);
      setStatus(t("eval.started"));
      await loadRuns();
      await loadDetail(payload.run_id);
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    }
  }

  function toggleSource(source, checked) {
    setForm((current) => ({
      ...current,
      sources: checked ? [...current.sources, source] : current.sources.filter((item) => item !== source),
    }));
  }

  return (
    <section id="evaluationView" className="app-view evaluation-view is-active">
      <div className="view-header">
        <div>
          <p className="eyebrow">Evaluation</p>
          <h1>{t("eval.title")}</h1>
          <p className="view-lead">{t("eval.lead")}</p>
        </div>
        <button className="ghost-button" type="button" onClick={loadRuns} disabled={isLoading}>
          {isLoading ? t("eval.refreshing") : t("eval.refresh")}
        </button>
      </div>

      <form className="evaluation-create" onSubmit={handleSubmit}>
        <div className="evaluation-main-input">
          <label className="auth-field">
            <span>{t("eval.runName")}</span>
            <input value={form.name} onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))} placeholder={t("eval.runNamePlaceholder")} />
          </label>
          <label className="auth-field">
            <span>{t("eval.queries")}</span>
            <textarea value={form.queryText} onChange={(event) => setForm((current) => ({ ...current, queryText: event.target.value }))} placeholder={t("eval.queriesPlaceholder")} required />
          </label>
        </div>
        <div className="evaluation-settings">
          <label className="auth-field">
            <span>{t("eval.mode")}</span>
            <select value={form.searchMode} onChange={(event) => setForm((current) => ({ ...current, searchMode: event.target.value }))}>
              <option value="auto">{t("eval.modeAuto")}</option>
              <option value="rules">{t("eval.modeRules")}</option>
              <option value="llm">{t("eval.modeLlm")}</option>
            </select>
          </label>
          <label className="auth-field">
            <span>{t("eval.limit")}</span>
            <input value={form.limit} onChange={(event) => setForm((current) => ({ ...current, limit: event.target.value }))} type="number" min="1" max="50" />
          </label>
          <label className="auth-field">
            <span>{t("eval.concurrency")}</span>
            <input value={form.concurrency} onChange={(event) => setForm((current) => ({ ...current, concurrency: event.target.value }))} type="number" min="1" max="8" />
          </label>
          <label className="auth-field">
            <span>{t("eval.year")}</span>
            <input value={form.year} onChange={(event) => setForm((current) => ({ ...current, year: event.target.value }))} placeholder="2022-2026" />
          </label>
          <div className="evaluation-source-list" aria-label={t("eval.sources")}>
            {sourceOptions.map((source) => (
              <label key={source}>
                <input type="checkbox" checked={form.sources.includes(source)} onChange={(event) => toggleSource(source, event.target.checked)} />
                <span>{source}</span>
              </label>
            ))}
          </div>
          <label className="evaluation-check">
            <input type="checkbox" checked={form.includeNeedsReview} onChange={(event) => setForm((current) => ({ ...current, includeNeedsReview: event.target.checked }))} />
            <span>{t("eval.includeNeedsReview")}</span>
          </label>
          <button className="primary-button" type="submit">{t("eval.start")}</button>
        </div>
      </form>

      {status ? <p className={`analysis-status ${isError ? "error" : ""}`}>{status}</p> : null}

      <div className="evaluation-layout">
        <aside className="evaluation-run-list" aria-label={t("eval.runs")}>
          {runs.length ? runs.map((run) => (
            <button className={`evaluation-run-item ${selectedRunId === run.id ? "is-active" : ""}`} type="button" key={run.id} onClick={() => setSelectedRunId(run.id)}>
              <span className={`history-status history-status--${run.status === "done" ? "done" : run.status === "error" ? "error" : "running"}`}>{run.status}</span>
              <strong>{run.name}</strong>
              <small>{run.completed_count || 0} / {run.total_count || 0} · {formatDateTime(run.created_at)}</small>
            </button>
          )) : <div className="history-empty">{t("eval.empty")}</div>}
        </aside>
        <EvaluationDetail detail={detail} t={t} />
      </div>
    </section>
  );
}

function EvaluationDetail({ detail, t }) {
  const metrics = useMemo(() => summarizeRun(detail), [detail]);
  if (!detail) return <section className="evaluation-detail"><div className="history-empty">{t("eval.noSelection")}</div></section>;
  return (
    <section className="evaluation-detail">
      <div className="evaluation-detail-header">
        <div>
          <span className={`history-status history-status--${detail.status === "done" ? "done" : detail.status === "error" ? "error" : "running"}`}>{detail.status}</span>
          <h2>{detail.name}</h2>
          <p>{detail.completed_count || 0} / {detail.total_count || 0} · {formatDateTime(detail.created_at)}</p>
        </div>
        <a className="ghost-button" href={evaluationExportUrl(detail.id)}>{t("eval.exportCsv")}</a>
      </div>
      <div className="admin-metrics evaluation-metrics">
        <Metric label={t("eval.avgTotal")} value={formatMs(metrics.avgTotal)} />
        <Metric label={t("eval.doneCount")} value={metrics.done} />
        <Metric label={t("eval.errorCount")} value={metrics.error} />
      </div>
      <div className="admin-table-scroll">
        <table className="analysis-table evaluation-table">
          <thead>
            <tr>
              <th>{t("eval.query")}</th>
              <th>{t("eval.status")}</th>
              <th>{t("eval.total")}</th>
              <th>{t("eval.stages")}</th>
              <th>{t("eval.counts")}</th>
              <th>{t("eval.firstResults")}</th>
            </tr>
          </thead>
          <tbody>
            {(detail.items || []).map((item) => <EvaluationRow item={item} t={t} key={item.id} />)}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EvaluationRow({ item, t }) {
  const result = item.result || {};
  const timings = result.timings || {};
  const references = [...(result.qualified_references || []), ...(result.needs_review_references || [])].slice(0, 3);
  return (
    <tr>
      <th><strong>{item.query}</strong>{item.error ? <span>{item.error}</span> : null}</th>
      <td><span className={`history-status history-status--${item.status === "done" ? "done" : item.status === "error" ? "error" : "running"}`}>{item.status}</span></td>
      <td>{formatMs(item.total_duration_ms)}</td>
      <td>
        <div className="evaluation-stage-list">
          <span>{t("eval.planning")}: {formatSeconds(timings.planning_seconds)}</span>
          <span>{t("eval.recall")}: {formatSeconds(timings.recall_seconds)}</span>
          <span>{t("eval.screening")}: {formatSeconds(timings.screening_seconds)}</span>
          <span>{t("eval.verification")}: {formatSeconds(timings.verification_seconds)}</span>
        </div>
      </td>
      <td>{t("eval.countValue", { qualified: (result.qualified_references || []).length, needsReview: (result.needs_review_references || []).length, rejected: result.rejected_count || 0, raw: result.raw_count || 0 })}</td>
      <td>
        <ol className="evaluation-result-list">
          {references.map((reference, index) => <li key={`${reference.title || "item"}-${index}`}>{reference.title || t("candidate.untitled")}</li>)}
        </ol>
      </td>
    </tr>
  );
}

function Metric({ label, value }) {
  return (
    <div className="history-meta-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function summarizeRun(detail) {
  const items = detail?.items || [];
  const doneItems = items.filter((item) => item.status === "done");
  const totals = doneItems.map((item) => Number(item.total_duration_ms || 0)).filter(Boolean);
  return {
    done: doneItems.length,
    error: items.filter((item) => item.status === "error").length,
    avgTotal: totals.length ? Math.round(totals.reduce((sum, value) => sum + value, 0) / totals.length) : 0,
  };
}

function formatSeconds(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(2)}s`;
}

function formatMs(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "-";
  return number >= 1000 ? `${(number / 1000).toFixed(2)}s` : `${number}ms`;
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}
