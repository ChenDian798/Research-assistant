import { assertUploadSize, buildLiteratureUserContext } from "./formatters.js";

const appBasePath = (window.location.pathname.match(/^\/v\d+(?=\/|$)/) || [""])[0];

export function apiPath(path) {
  return `${appBasePath}${path}`;
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
  const response = await fetch(apiPath("/api/literature-search"), {
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
  const response = await fetch(apiPath("/api/novelty-check"), {
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
  const response = await fetch(apiPath("/api/history"), { cache: "no-store" });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return Array.isArray(data.history) ? data.history : [];
}

export async function fetchHistoryEntry(historyId) {
  const response = await fetch(apiPath(`/api/history/${encodeURIComponent(historyId)}`), { cache: "no-store" });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

export async function deleteHistoryEntry(historyId) {
  const response = await fetch(apiPath(`/api/history/${encodeURIComponent(historyId)}`), { method: "DELETE" });
  const data = await readJsonResponse(response);
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function normalizeOutputLanguage(language) {
  return language === "en" ? "en" : "zh";
}

export async function submitLiteratureAnalysis({ topic = "literature-analysis", references = [], finalReport = "", historySource = "direct", historyId = "", outputLanguage = "zh" } = {}) {
  return fetch(apiPath("/api/literature-analysis"), {
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
  return fetch(apiPath("/api/literature-analysis/pdf"), { method: "POST", body: formData });
}

export async function waitForJob(basePath, jobId, updateStatus, t) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 10 * 60 * 1000) {
    await sleep(1000);
    const response = await fetch(apiPath(`${basePath}/${jobId}`), { cache: "no-store" });
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
