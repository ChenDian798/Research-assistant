import { normalizeLiteratureSummary } from "./formatters.js";

export const allowedViews = ["home", "direct", "standaloneSearch", "novelty", "search", "results", "history"];

const legacyViewMap = { literature: "home" };

export const defaultSearchForm = {
  query: "",
  searchMode: "auto",
  year: "",
  limit: "5",
  sources: ["arxiv", "pubmed"],
  includeNeedsReview: true,
  appendAnnotationRecord: true,
};

export const defaultNoveltyForm = {
  innovationText: "",
  searchMode: "auto",
  year: "",
  sources: ["arxiv", "pubmed", "semantic"],
  includeFilteredReferences: false,
};

export function viewFromHash(hash = window.location.hash) {
  const value = hash.replace("#", "");
  return legacyViewMap[value] || (allowedViews.includes(value) ? value : "home");
}

export function historySearchCandidates(entry) {
  const result = entry?.result || {};
  const qualified = Array.isArray(result.qualified_references) ? result.qualified_references : [];
  const needsReview = Array.isArray(result.needs_review_references) ? result.needs_review_references : [];
  const rejected = Array.isArray(result.rejected_references) ? result.rejected_references : [];
  return [...qualified, ...needsReview, ...rejected].map((reference, index) => ({
    ...reference,
    candidate_group: index < qualified.length ? "qualified" : index < qualified.length + needsReview.length ? "needs_review" : "rejected",
    candidate_id: reference.dedupe_key || reference.source || reference.doi || reference.title || `candidate-${index}`,
  }));
}

export function hasStoredAnalysisResult(entry) {
  const result = entry?.result || {};
  const rows = Array.isArray(result.rows) ? result.rows : [];
  const reviewNeededDocuments = Array.isArray(entry?.request?.review_needed_documents)
    ? entry.request.review_needed_documents
    : [];
  return Boolean(rows.length || normalizeLiteratureSummary(result.summary) || reviewNeededDocuments.length);
}

export function storedAnalysisResult(entry, fallbackTopic = "literature-analysis") {
  const request = entry?.request || {};
  const result = entry?.result || {};
  const rows = Array.isArray(result.rows) ? result.rows : [];
  const references = Array.isArray(result.references) ? result.references : [];
  const requestReferences = Array.isArray(request.references) ? request.references : [];
  const reviewNeededDocuments = Array.isArray(request.review_needed_documents) ? request.review_needed_documents : [];
  return {
    rows,
    summary: normalizeLiteratureSummary(result.summary),
    topic: request.topic || fallbackTopic,
    displayReferences: references.length ? references : requestReferences,
    reviewNeededDocuments,
  };
}

export function isActiveHistoryEntry(entry) {
  if (!entry) return false;
  return ["queued", "running"].includes(entry.status) || ["queued", "running"].includes(entry.analysis?.status);
}

export function isHistorySummaryOnly(entry) {
  return Boolean(entry?.is_summary);
}

export function mergeHistorySummaryIntoDetail(detail, summary) {
  if (!summary) return detail || null;
  if (!detail || detail.id !== summary.id) return summary;
  const detailIsSummary = isHistorySummaryOnly(detail);
  const merged = {
    ...detail,
    ...summary,
    is_summary: detailIsSummary,
    request: !detailIsSummary && nonEmptyObject(detail.request) ? detail.request : (summary.request || {}),
    result: !detailIsSummary && nonEmptyObject(detail.result) ? detail.result : (summary.result || {}),
    counts: summary.counts || detail.counts || {},
  };
  if (detail.analysis || summary.analysis) {
    merged.analysis = mergeHistoryAnalysisSummary(detail.analysis, summary.analysis);
  }
  return merged;
}

function mergeHistoryAnalysisSummary(detailAnalysis, summaryAnalysis) {
  if (!summaryAnalysis) return detailAnalysis || null;
  if (!detailAnalysis) return summaryAnalysis;
  const detailIsSummary = isHistorySummaryOnly(detailAnalysis);
  return {
    ...detailAnalysis,
    ...summaryAnalysis,
    is_summary: detailIsSummary,
    request: !detailIsSummary && nonEmptyObject(detailAnalysis.request) ? detailAnalysis.request : (summaryAnalysis.request || {}),
    result: !detailIsSummary && nonEmptyObject(detailAnalysis.result) ? detailAnalysis.result : (summaryAnalysis.result || {}),
    counts: summaryAnalysis.counts || detailAnalysis.counts || {},
  };
}

function nonEmptyObject(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length);
}
