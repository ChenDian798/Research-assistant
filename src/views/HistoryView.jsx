import { useEffect, useState } from "react";
import AnalysisTable from "../components/AnalysisTable.jsx";
import LiteratureSummary from "../components/LiteratureSummary.jsx";
import LoadingState from "../components/LoadingState.jsx";
import { normalizeLiteratureSummary } from "../lib/formatters.js";

export default function HistoryView({
  entries,
  selectedEntry,
  isLoading,
  errorMessage,
  onRefresh,
  onSelectEntry,
  onDeleteEntry,
  onResubmitEntry,
  onContinueEntry,
  t,
}) {
  const displayEntries = mergeHistoryEntries(entries);
  const currentEntry = selectedVisibleEntry(selectedEntry, displayEntries) || displayEntries[0] || null;

  return (
    <section id="historyView" className="app-view history-view is-active">
      <div className="view-header">
        <div>
          <p className="eyebrow">History</p>
          <h1>{t("history.title")}</h1>
          <p className="view-lead">{t("history.lead")}</p>
        </div>
        <button className="ghost-button" type="button" onClick={onRefresh} disabled={isLoading}>
          {isLoading ? t("history.refreshing") : t("history.refresh")}
        </button>
      </div>

      {errorMessage ? <p className="history-error" role="alert">{errorMessage}</p> : null}

      <div className="history-layout">
        <aside className="history-list-panel panel" aria-label={t("history.listAria")}>
          {displayEntries.length ? (
            <ul className="history-list">
              {displayEntries.map((entry) => (
                <li key={entry.id}>
                  <div className={`history-item-shell ${currentEntry?.id === entry.id ? "is-active" : ""}`}>
                  <button
                    className="history-item"
                    type="button"
                    onClick={() => onSelectEntry(entry.id)}
                  >
                    <span className={`history-status history-status--${statusTone(entry)}`}>{statusLabel(entry, t)}</span>
                    <strong>{displayHistoryTitle(entry, t)}</strong>
                    <small>{kindLabel(entry.kind, t)} · {formatDateTime(entry.created_at)}</small>
                  </button>
                  <button
                    className="history-delete-button"
                    type="button"
                    onClick={() => onDeleteEntry(entry)}
                    aria-label={t("history.deleteAria", { title: displayHistoryTitle(entry, t) })}
                    title={t("history.delete")}
                  >
                    <span className="history-delete-glyph" aria-hidden="true">x</span>
                    <span aria-hidden="true">×</span>
                  </button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="history-empty">{t("history.empty")}</div>
          )}
        </aside>

        <section className="history-detail-panel panel">
          {currentEntry ? (
            <HistoryDetail entry={currentEntry} onResubmitEntry={onResubmitEntry} onContinueEntry={onContinueEntry} t={t} />
          ) : (
            <div className="history-empty">{t("history.emptyDetail")}</div>
          )}
        </section>
      </div>
    </section>
  );
}

function selectedVisibleEntry(selectedEntry, entries) {
  if (!selectedEntry) return null;
  const visibleEntry = entries.find((entry) => entry.id === selectedEntry.id) ||
    entries.find((entry) => entry.title === selectedEntry.title && (entry.kind === "search_flow" || entry.kind === "literature_search"));
  if (!visibleEntry) return selectedEntry;
  return {
    ...visibleEntry,
    ...selectedEntry,
    kind: visibleEntry.kind || selectedEntry.kind,
    status: visibleEntry.status || selectedEntry.status,
    stage: visibleEntry.stage || selectedEntry.stage,
    job_id: visibleEntry.job_id || selectedEntry.job_id,
    analysis: visibleEntry.analysis || selectedEntry.analysis,
  };
}

function mergeHistoryEntries(entries) {
  const analysisByTitle = new Map();
  entries.forEach((entry) => {
    if (entry?.kind === "search_analysis" && entry.title) {
      analysisByTitle.set(entry.title, entry);
    }
  });
  const merged = [];
  const mergedAnalysisIds = new Set();
  entries.forEach((entry) => {
    if (!entry || entry.kind === "search_analysis") return;
    if ((entry.kind === "literature_search" || entry.kind === "search_flow") && entry.title && analysisByTitle.has(entry.title)) {
      const analysisEntry = analysisByTitle.get(entry.title);
      mergedAnalysisIds.add(analysisEntry.id);
      merged.push({
        ...entry,
        kind: "search_flow",
        status: analysisEntry.status || entry.status,
        stage: analysisEntry.stage || entry.stage,
        updated_at: analysisEntry.updated_at || entry.updated_at,
        job_id: analysisEntry.job_id || entry.job_id,
        analysis: {
          id: analysisEntry.id,
          status: analysisEntry.status,
          stage: analysisEntry.stage,
          job_id: analysisEntry.job_id,
          request: analysisEntry.request || {},
          result: analysisEntry.result || {},
          counts: analysisEntry.counts || {},
          error: analysisEntry.error || "",
        },
      });
      return;
    }
    merged.push(entry);
  });
  entries.forEach((entry) => {
    if (entry?.kind === "search_analysis" && !mergedAnalysisIds.has(entry.id)) merged.push(entry);
  });
  return merged;
}

function HistoryDetail({ entry, onResubmitEntry, onContinueEntry, t }) {
  const isAnalysis = entry.kind === "direct_analysis" || entry.kind === "search_analysis";
  const isSearch = entry.kind === "literature_search" || entry.kind === "search_flow";
  const request = entry.request || {};
  const result = entry.result || {};
  const references = Array.isArray(request.references) ? request.references : [];
  const resultReferences = Array.isArray(result.references) ? result.references : [];
  const reviewNeededDocuments = Array.isArray(request.review_needed_documents)
    ? request.review_needed_documents
    : (Array.isArray(result.review_needed_documents) ? result.review_needed_documents : []);
  const rows = Array.isArray(result.rows) ? result.rows : [];
  const summary = normalizeLiteratureSummary(result.summary);
  const flowAnalysis = entry.analysis && typeof entry.analysis === "object" ? entry.analysis : null;
  const flowAnalysisResult = flowAnalysis?.result || {};
  const flowAnalysisRows = Array.isArray(flowAnalysisResult.rows) ? flowAnalysisResult.rows : [];
  const flowAnalysisSummary = normalizeLiteratureSummary(flowAnalysisResult.summary);
  const flowAnalysisRequest = flowAnalysis?.request || {};
  const flowAnalysisReferences = Array.isArray(flowAnalysisRequest.references) ? flowAnalysisRequest.references : [];
  const flowAnalysisResultReferences = Array.isArray(flowAnalysisResult.references) ? flowAnalysisResult.references : [];
  const flowAnalysisReviewNeededDocuments = Array.isArray(flowAnalysisResult.review_needed_documents)
    ? flowAnalysisResult.review_needed_documents
    : [];
  const flowAnalysisMode = historyAnalysisMode(
    flowAnalysis?.status,
    flowAnalysisResult,
    flowAnalysisReviewNeededDocuments,
    flowAnalysis?.stage || entry.stage,
  );
  const mode = historyAnalysisMode(entry.status, result, reviewNeededDocuments, entry.stage);
  const searchMode = isSearch ? searchHistoryMode(entry) : mode;
  const isDirectAnalysisError = entry.kind === "direct_analysis" && (mode === "error" || mode === "interrupted");

  return (
    <div className="history-detail">
      <div className="history-detail-header">
        <div>
          <span className={`history-status history-status--${statusTone(entry)}`}>{statusLabel(entry, t)}</span>
          <h2>{displayHistoryTitle(entry, t)}</h2>
          <p>{kindLabel(entry.kind, t)} · {formatDateTime(entry.created_at)}</p>
        </div>
      </div>

      {entry.stage ? <p className="history-stage">{translateStage(entry.stage, t)}</p> : null}
      {entry.error ? <p className="history-error" role="alert">{entry.error}</p> : null}
      {isDirectAnalysisError ? (
        <div className="history-actions">
          <span>{t("history.directInterruptedHint")}</span>
        </div>
      ) : entry.status === "error" ? (
        <div className="history-actions">
          <button className="primary-button" type="button" onClick={() => onResubmitEntry(entry)}>
            {t("history.resubmit")}
          </button>
          <span>{resubmitHint(entry, t)}</span>
        </div>
      ) : null}
      {isSearch && entry.status !== "error" && searchMode === "done" && flowAnalysisMode !== "loading" ? (
        <div className="history-actions">
          <button className="primary-button" type="button" onClick={() => onContinueEntry(entry)}>
            {t("history.continueFlow")}
          </button>
          <span>{t("history.continueFlowHint")}</span>
        </div>
      ) : null}

      {isAnalysis ? (
        <>
          <div className="history-meta-grid">
            <MetaItem label={t("history.topic")} value={request.topic || entry.title} />
            <MetaItem label={t("history.referenceCount")} value={String(request.reference_count ?? references.length)} />
            <MetaItem label={t("history.jobId")} value={entry.job_id || "-"} />
          </div>
          {mode === "loading" ? (
            <LoadingState
              title={t("table.loadingTitle")}
              message={references.length ? t("table.loadingWithCount", { count: references.length }) : t("table.loadingDefault")}
            />
          ) : (
            <>
              <LiteratureSummary summary={summary} t={t} />
              <AnalysisTable
                rows={rows}
                displayReferences={resultReferences.length ? resultReferences : references}
                reviewNeededDocuments={reviewNeededDocuments}
                mode={mode}
                errorMessage={entry.error || ""}
                t={t}
              />
            </>
          )}
        </>
      ) : null}

      {isSearch ? <SearchHistoryDetail entry={entry} mode={searchMode} t={t} /> : null}
      {flowAnalysis ? (
        <div className="history-flow-analysis">
          <h3>{t("history.analysisSection")}</h3>
          <LiteratureSummary summary={flowAnalysisSummary} t={t} />
          <AnalysisTable
            rows={flowAnalysisRows}
            displayReferences={flowAnalysisResultReferences.length ? flowAnalysisResultReferences : flowAnalysisReferences}
            reviewNeededDocuments={flowAnalysisReviewNeededDocuments}
            mode={flowAnalysisMode}
            errorMessage={flowAnalysis.error || entry.error || ""}
            t={t}
          />
        </div>
      ) : null}
    </div>
  );
}

function historyAnalysisMode(status, result, reviewNeededDocuments = [], stage = "") {
  if (hasAnalysisResultContent(result, reviewNeededDocuments)) return "done";
  if (stage === "Task interrupted") return "interrupted";
  if (status === "running" || status === "queued") return "loading";
  if (status === "error") return "error";
  if (status === "done") return "done";
  return "idle";
}

function hasAnalysisResultContent(result, reviewNeededDocuments = []) {
  const rows = Array.isArray(result?.rows) ? result.rows : [];
  return Boolean(rows.length || normalizeLiteratureSummary(result?.summary) || reviewNeededDocuments.length);
}

function searchHistoryMode(entry) {
  const result = entry?.result || {};
  const qualified = Array.isArray(result.qualified_references) ? result.qualified_references : [];
  const needsReview = Array.isArray(result.needs_review_references) ? result.needs_review_references : [];
  if (qualified.length || needsReview.length || result.status === "done" || entry?.stage === "Search complete") {
    return "done";
  }
  if (entry?.status === "error") return "error";
  if (entry?.status === "running" || entry?.status === "queued") return "loading";
  return entry?.status === "done" ? "done" : "idle";
}

function resubmitHint(entry, t) {
  if (entry.kind === "direct_analysis" && entry.request?.file_count) {
    return t("history.resubmitFilesHint");
  }
  if (entry.kind === "literature_search" || entry.kind === "search_flow") {
    return searchHistoryMode(entry) === "done" ? t("history.continueFlowHint") : t("history.resubmitSearchHint");
  }
  if (entry.kind === "search_analysis") return t("history.resubmitAnalysisHint");
  return t("history.resubmitDefaultHint");
}

function SearchHistoryDetail({ entry, mode, t }) {
  const [candidatesCollapsed, setCandidatesCollapsed] = useState(false);
  const request = entry.request || {};
  const result = entry.result || {};
  const qualified = Array.isArray(result.qualified_references) ? result.qualified_references : [];
  const needsReview = Array.isArray(result.needs_review_references) ? result.needs_review_references : [];
  const candidates = [...qualified, ...needsReview];

  useEffect(() => {
    setCandidatesCollapsed(false);
  }, [entry.id]);

  return (
    <>
      <div className="history-meta-grid">
        <MetaItem label={t("history.query")} value={request.query || result.query || entry.title} />
        <MetaItem label={t("history.sources")} value={request.sources || "-"} />
        <MetaItem label={t("history.searchMode")} value={request.search_mode || result.search_mode || "-"} />
        <MetaItem label={t("history.candidateCount")} value={String(candidates.length)} />
      </div>
      {mode === "loading" ? (
        <LoadingState
          title={t("history.searchLoadingTitle")}
          message={t("history.searchLoadingBody")}
        />
      ) : candidates.length ? (
        <div className="history-candidate-section">
          <div className="history-candidate-toolbar">
            <div>
              <span>{t("history.candidatesTitle")}</span>
              <strong>{t("history.candidatesCount", { count: candidates.length })}</strong>
            </div>
            <button
              className={`history-collapse-button ${candidatesCollapsed ? "is-collapsed" : ""}`}
              type="button"
              aria-expanded={!candidatesCollapsed}
              onClick={() => setCandidatesCollapsed((collapsed) => !collapsed)}
            >
              <span className="history-collapse-icon" aria-hidden="true" />
              {candidatesCollapsed ? t("history.expandCandidates") : t("history.collapseCandidates")}
            </button>
          </div>
          {candidatesCollapsed ? (
            <div className="history-candidate-collapsed">
              {t("history.candidatesCollapsed", { count: candidates.length })}
            </div>
          ) : (
            <ol className="history-candidate-list">
              {candidates.slice(0, 30).map((candidate, index) => (
                <li key={`${candidate.doi || candidate.source || candidate.title || "candidate"}-${index}`}>
                  <strong>{candidate.title || t("candidate.untitled")}</strong>
                  <small>{candidate.doi || candidate.pmid || candidate.arxiv_id || candidate.source || t("reference.noStableId")}</small>
                </li>
              ))}
            </ol>
          )}
        </div>
      ) : (
        <div className="history-empty">{t("history.noCandidates")}</div>
      )}
    </>
  );
}

function MetaItem({ label, value }) {
  return (
    <div className="history-meta-item">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function displayHistoryTitle(entry, t) {
  const rawTitle = String(entry?.title || "").trim();
  if (!entry || !["direct_analysis", "search_analysis"].includes(entry.kind)) {
    return rawTitle || t("history.untitled");
  }
  if (/^(直接分析|检索流分析)\s*·/.test(rawTitle)) return rawTitle;

  const request = entry.request || {};
  const counts = entry.counts || {};
  const references = Array.isArray(request.references) ? request.references : [];
  const referenceCount = firstPositiveInt(request.reference_count, counts.references, references.length);
  const itemText = referenceCount ? `${referenceCount}篇` : "资料";
  const prefix = entry.source === "search" || entry.kind === "search_analysis" ? "检索流分析" : "直接分析";
  const subject = historySubject(rawTitle, references);
  return `${prefix} · ${itemText} · ${subject}`;
}

function firstPositiveInt(...values) {
  for (const value of values) {
    const number = Number.parseInt(value, 10);
    if (number > 0) return number;
  }
  return 0;
}

function historySubject(title, references) {
  const candidates = [
    String(title || "").trim(),
    ...references.slice(0, 4).map((reference) => String(reference?.title || reference?.source || "").trim()),
  ].filter(Boolean);
  const combined = candidates.join(" ");
  const known = knownHistorySubject(combined);
  if (known) return known;
  for (const candidate of candidates) {
    const cleaned = cleanHistorySubject(candidate);
    if (cleaned) return cleaned;
  }
  return "文献分析";
}

function knownHistorySubject(text) {
  const normalized = String(text || "")
    .toLowerCase()
    .replace(/\.(pdf|docx?)\b/gi, " ")
    .replace(/\b(arxiv|pubmed|pmid|doi)\b/gi, " ")
    .replace(/\b10\.\d{4,9}\/\S+/gi, " ")
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/[_\-.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const checks = [
    [["stroke", "segmentation"], "卒中分割"],
    [["ischemic stroke"], "缺血性卒中"],
    [["medical imaging", "segmentation"], "医学影像分割"],
    [["deep learning"], "深度学习"],
    [["machine learning"], "机器学习"],
    [["ct", "segmentation"], "CT分割"],
    [["mri", "segmentation"], "MRI分割"],
  ];
  const match = checks.find(([needles]) => needles.every((needle) => normalized.includes(needle)));
  return match ? match[1] : "";
}

function cleanHistorySubject(text) {
  let cleaned = String(text || "").trim();
  if (!cleaned) return "";
  cleaned = cleaned
    .replace(/\.(pdf|docx?)\b/gi, " ")
    .replace(/\b(arxiv|pubmed|pmid|doi)\b/gi, " ")
    .replace(/\b10\.\d{4,9}\/\S+/gi, " ")
    .replace(/https?:\/\/\S+/gi, " ")
    .replace(/(^|\s)\d{1,3}[_\-\s]+/g, " ")
    .replace(/[_\-]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/^[ ·,;；，。]+|[ ·,;；，。]+$/g, "");
  const sentence = cleaned.split(/[。！？!?；;\n\r]/)[0]?.trim();
  if (sentence) cleaned = sentence;
  const generic = new Set([
    "current research",
    "literature analysis",
    "literature-analysis",
    "user provided literature links and pdf analysis",
    "user-provided literature links and pdf analysis",
  ]);
  return generic.has(cleaned.toLowerCase()) ? "" : cleaned.slice(0, 28).trim();
}

function kindLabel(kind, t) {
  const key = {
    direct_analysis: "history.kind.direct",
    search_analysis: "history.kind.searchAnalysis",
    literature_search: "history.kind.search",
    search_flow: "history.kind.searchFlow",
  }[kind] || "history.kind.unknown";
  return t(key);
}

function statusTone(entry) {
  if (isAnalysisEntryReady(entry)) return "done";
  if (isSearchResultReady(entry)) return "searched";
  return entry?.status || "idle";
}

function statusLabel(entry, t) {
  if (isAnalysisEntryReady(entry)) return t("history.status.done");
  if (isSearchResultReady(entry)) return t("history.status.searched");
  const status = entry?.status;
  const key = {
    queued: "history.status.queued",
    running: "history.status.running",
    done: "history.status.done",
    error: "history.status.error",
  }[status] || "history.status.idle";
  return t(key);
}

function isAnalysisEntryReady(entry) {
  if (!entry) return false;
  if (["direct_analysis", "search_analysis"].includes(entry.kind)) {
    return hasAnalysisResultContent(entry.result || {}, entry.request?.review_needed_documents || []);
  }
  const analysis = entry.analysis && typeof entry.analysis === "object" ? entry.analysis : null;
  if (!analysis) return false;
  const analysisResult = analysis.result || {};
  const reviewNeededDocuments = Array.isArray(analysisResult.review_needed_documents)
    ? analysisResult.review_needed_documents
    : [];
  return hasAnalysisResultContent(analysisResult, reviewNeededDocuments);
}

function isSearchResultReady(entry) {
  if (!entry || entry.status !== "done") return false;
  if (!["literature_search", "search_flow"].includes(entry.kind)) return false;
  return !(entry.analysis && typeof entry.analysis === "object");
}

function translateStage(stage, t) {
  const stageMap = {
    "Starting literature analysis...": t("job.starting"),
    "Resolving DOI metadata...": t("job.resolvingDoi"),
    "Running LLM literature analysis...": t("job.runningAnalysis"),
    "Analysis complete": t("history.stageComplete"),
    "Searching literature...": t("history.searchLoadingTitle"),
    "Search complete": t("history.searchComplete"),
    "Task interrupted": t("history.taskInterrupted"),
  };
  return stageMap[stage] || stage;
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}
