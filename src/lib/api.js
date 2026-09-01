import { assertUploadSize, buildLiteratureUserContext } from "./formatters.js";

const appBasePath = (window.location.pathname.match(/^\/v\d+(?=\/|$)/) || [""])[0];
let csrfToken = null;

export function apiPath(path) {
  return `${appBasePath}${path}`;
}

export async function apiFetch(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const requestOptions = { ...options, credentials: "same-origin" };
  const csrfExempt = ["/api/auth/login", "/api/auth/register"].includes(path);
  if (!csrfExempt && !["GET", "HEAD", "OPTIONS"].includes(method)) {
    if (!csrfToken) {
      const csrfResponse = await fetch(apiPath("/api/auth/csrf"), { credentials: "same-origin", cache: "no-store" });
      const csrfPayload = await readJsonResponse(csrfResponse);
      if (!csrfResponse.ok || !csrfPayload.csrf_token) {
        const error = new Error(csrfPayload.error || "请先登录后再继续。");
        error.status = csrfResponse.status;
        throw error;
      }
      csrfToken = csrfPayload.csrf_token;
    }
    requestOptions.headers = { ...(options.headers || {}), "X-CSRF-Token": csrfToken };
  }
  const response = await fetch(apiPath(path), requestOptions);
  if (response.status === 401) csrfToken = null;
  return response;
}

export async function fetchCurrentUser() {
  const response = await apiFetch("/api/auth/me", { cache: "no-store" });
  if (response.status === 401) return null;
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload.user || null;
}

export async function login(email, password) {
  const response = await apiFetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  csrfToken = null;
  return payload.user || null;
}

export async function register({ email, password, displayName = "" }) {
  const response = await apiFetch("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, display_name: displayName }),
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  csrfToken = null;
  return payload.user || null;
}

export async function logout() {
  const response = await apiFetch("/api/auth/logout", { method: "POST" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  csrfToken = null;
}

export async function fetchAdminUsers() {
  const response = await apiFetch("/api/admin/users", { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return Array.isArray(payload.users) ? payload.users : [];
}

export async function fetchAdminHistory({ limit = 100, ownerUserId = "", ownerKeyword = "", kind = "", status = "" } = {}) {
  const params = new URLSearchParams();
  params.set("limit", String(limit || 100));
  if (ownerUserId) params.set("owner_user_id", ownerUserId);
  if (ownerKeyword) params.set("owner_keyword", ownerKeyword);
  if (kind) params.set("kind", kind);
  if (status) params.set("status", status);
  const response = await apiFetch(`/api/admin/history?${params.toString()}`, { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return Array.isArray(payload.history) ? payload.history : [];
}

export async function fetchAdminHistoryDetail(historyId) {
  const response = await apiFetch(`/api/admin/history/${encodeURIComponent(historyId)}`, { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

export async function fetchAdminMetrics(days = 7) {
  const params = new URLSearchParams({ days: String(days || 7) });
  const response = await apiFetch(`/api/admin/metrics?${params.toString()}`, { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

export async function fetchAdminAuditLogs(limit = 30, actorUserId = "") {
  const params = new URLSearchParams({ limit: String(limit || 30) });
  if (actorUserId) params.set("actor_user_id", actorUserId);
  const response = await apiFetch(`/api/admin/audit-logs?${params.toString()}`, { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return Array.isArray(payload.audit_logs) ? payload.audit_logs : [];
}

export async function createAdminUser({ email, displayName = "", password = "", role = "user" }) {
  const response = await apiFetch("/api/admin/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, display_name: displayName, password, role }),
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

export async function updateAdminUser(userId, patch) {
  const response = await apiFetch(`/api/admin/users/${encodeURIComponent(userId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload.user || null;
}

export async function deleteAdminUser(userId) {
  const response = await apiFetch(`/api/admin/users/${encodeURIComponent(userId)}`, { method: "DELETE" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

export async function createSearchEvaluation(payload) {
  const response = await apiFetch("/api/admin/evaluations/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

export async function fetchEvaluations() {
  const response = await apiFetch("/api/admin/evaluations", { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return Array.isArray(payload.runs) ? payload.runs : [];
}

export async function fetchEvaluationDetail(runId) {
  const response = await apiFetch(`/api/admin/evaluations/${encodeURIComponent(runId)}`, { cache: "no-store" });
  const payload = await readJsonResponse(response);
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

export function evaluationExportUrl(runId) {
  return apiPath(`/api/admin/evaluations/${encodeURIComponent(runId)}/export.csv`);
}

export async function readJsonResponse(response) {
  const responseText = await response.text();
  if (!responseText) return {};
  try {
    return JSON.parse(responseText);
  } catch (error) {
    if (response.status === 413) {
      throw new Error("上传文件过大，服务器拒绝接收。请减少文件大小或分批上传。");
    }
    const plainText = responseText.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
    throw new Error(`服务返回了无法解析的数据：${plainText.slice(0, 160) || responseText.slice(0, 160)}`);
  }
}

export async function submitLiteratureSearchRequest(payload) {
  const response = await apiFetch("/api/literature-search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    const error = new Error(data.error || `HTTP ${response.status}`);
    error.payload = data;
    throw error;
  }
  return data;
}

export async function submitNoveltyCheckRequest(payload) {
  const response = await apiFetch("/api/novelty-check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await readJsonResponse(response);
  if (!response.ok) {
    const error = new Error(data.error || `HTTP ${response.status}`);
    error.payload = data;
    throw error;
  }
  return data;
}

export async function fetchHistoryEntries() {
  const response = await apiFetch("/api/history", { cache: "no-store" });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return Array.isArray(data.history) ? data.history : [];
}

export async function fetchHistoryEntry(historyId) {
  const response = await apiFetch(`/api/history/${encodeURIComponent(historyId)}`, { cache: "no-store" });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

export async function deleteHistoryEntry(historyId) {
  const response = await apiFetch(`/api/history/${encodeURIComponent(historyId)}`, { method: "DELETE" });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

export async function submitReferenceFeedback({ historyId = "", referenceKey = "", feedbackKind = "reference_relevance", vote = "", reference = {} } = {}) {
  const response = await apiFetch("/api/reference-feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      history_id: historyId,
      reference_key: referenceKey,
      feedback_kind: feedbackKind,
      vote,
      reference: {
        title: reference.title || "",
        source_label: reference.source_label || "",
        retrieved_from: reference.retrieved_from || "",
        doi: reference.doi || "",
        pmid: reference.pmid || "",
        arxiv_id: reference.arxiv_id || "",
        candidate_id: reference.candidate_id || "",
        dedupe_key: reference.dedupe_key || "",
        source: reference.source || "",
      },
    }),
  });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function normalizeOutputLanguage(language) {
  return language === "en" ? "en" : "zh";
}

export async function submitLiteratureAnalysis({ topic = "literature-analysis", references = [], finalReport = "", historySource = "direct", historyId = "", outputLanguage = "zh" } = {}) {
  return apiFetch("/api/literature-analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      references,
      final_report: finalReport,
      history_source: historySource,
      history_id: historyId,
      output_language: normalizeOutputLanguage(outputLanguage),
    }),
  });
}

export async function submitLinkLiteratureAnalysis(references, userContext = "", topic = "literature-analysis", analysisSource = "direct", historyId = "", outputLanguage = "zh") {
  const fallback = analysisSource === "search"
    ? "The user selected these references from the literature search flow for final literature analysis."
    : "The user provided DOI identifiers or literature links directly in the literature assistant.";
  return submitLiteratureAnalysis({
    topic,
    references,
    finalReport: buildLiteratureUserContext(userContext, fallback),
    historySource: analysisSource,
    historyId,
    outputLanguage,
  });
}

export async function submitCombinedLiteratureAnalysis(references, pdfFiles, userContext = "", topic = "literature-analysis", historySource = "direct", outputLanguage = "zh") {
  assertUploadSize(pdfFiles);
  const formData = new FormData();
  formData.append("topic", topic);
  formData.append("references", JSON.stringify(references));
  formData.append("user_context", userContext);
  formData.append("history_source", historySource);
  formData.append("output_language", normalizeOutputLanguage(outputLanguage));
  pdfFiles.forEach((file) => formData.append("pdf", file));
  return apiFetch("/api/literature-analysis/pdf", { method: "POST", body: formData });
}

export async function waitForJob(basePath, jobId, updateStatus, t) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 10 * 60 * 1000) {
    await sleep(1000);
    const response = await apiFetch(`${basePath}/${jobId}`, { cache: "no-store" });
    const payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.status === "done") return payload;
    if (payload.status === "error") throw new Error(payload.error || translate("job.failed", t));
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    updateStatus(translateJobStage(payload.stage, t) || translate("job.running", t, { elapsed }));
  }
  throw new Error(translate("job.timeout", t));
}

function translateJobStage(stage, t) {
  const text = String(stage || "").trim();
  const stageMap = {
    "Starting literature analysis...": translate("job.starting", t),
    "Resolving DOI metadata...": translate("job.resolvingDoi", t),
    "Running LLM literature analysis...": translate("job.runningAnalysis", t),
    "Searching literature...": translate("history.searchLoadingTitle", t),
    "Search complete": translate("history.searchComplete", t),
    "Planning novelty search...": translate("job.planningNovelty", t),
    "Assessing novelty overlap...": translate("job.assessingNovelty", t),
    "Novelty check complete": translate("job.noveltyComplete", t),
  };
  return stageMap[text] || text;
}

function translate(key, t, params) {
  if (typeof t === "function") return t(key, params);
  const fallback = {
    "job.failed": "任务运行失败。",
    "job.running": ({ elapsed }) => `运行中，已运行 ${elapsed} 秒...`,
    "job.timeout": "任务超过 10 分钟未完成，已停止等待。",
    "job.starting": "正在启动文献分析...",
    "job.resolvingDoi": "正在补全文献元数据...",
    "job.runningAnalysis": "正在运行文献分析...",
    "job.planningNovelty": "Planning novelty search...",
    "job.assessingNovelty": "Assessing novelty overlap...",
    "job.noveltyComplete": "Novelty check complete.",
  }[key];
  return typeof fallback === "function" ? fallback(params || {}) : fallback;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
