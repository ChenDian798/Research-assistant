const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const literatureForm = $("#literatureForm");
const doiInput = $("#doiInput");
const doiAnalyzeButton = $("#doiAnalyzeButton");
const searchAnalyzeButton = $("#searchAnalyzeButton");
const pdfInput = $("#pdfInput");
const pdfFileSummary = $("#pdfFileSummary");
const linkInputSummary = $("#linkInputSummary");
const clearPdfButton = $("#clearPdfButton");
const literatureStatus = $("#literatureStatus");
const doiAnalysisBody = $("#doiAnalysisBody");
const doiSummary = $("#doiSummary");
const literatureExportFormatWrap = $("#literatureExportFormatWrap");
const literatureExportFormat = $("#literatureExportFormat");
const literatureExportButton = $("#literatureExportButton");
const literatureSearchQuery = $("#literatureSearchQuery");
const literatureSearchYear = $("#literatureSearchYear");
const literatureSearchLimit = $("#literatureSearchLimit");
const includeNeedsReviewSearch = $("#includeNeedsReviewSearch");
const appendAnnotationRecordSearch = $("#appendAnnotationRecordSearch");
const literatureSearchButton = $("#literatureSearchButton");
const literatureSearchStatus = $("#literatureSearchStatus");
const searchCandidatePanel = $("#searchCandidatePanel");
const searchCandidateList = $("#searchCandidateList");
const searchCandidateMeta = $("#searchCandidateMeta");
const addSearchReferencesButton = $("#addSearchReferencesButton");
const stagedReferenceList = $("#stagedReferenceList");
const appViews = $$("[data-app-view]");
const viewButtons = $$("[data-view-target]");
const searchStepButtons = $$("[data-search-step-target]");
const searchSteps = $$("[data-search-step]");
const stepperItems = $$(".step[data-search-step-target]");

let latestDoiAnalysisRows = [];
let latestDoiAnalysisSummary = null;
let latestDoiTopic = "literature-analysis";
let activeAnalysisSource = "direct";
const analysisRunIds = { direct: 0, search: 0 };
const analysisResultsBySource = {
  direct: createEmptyAnalysisResult(),
  search: createEmptyAnalysisResult(),
};
let selectedPdfFiles = [];
let searchCandidateReferences = [];
let selectedSearchReferenceIds = new Set();
let stagedAnalysisReferences = [];
const appBasePath = (window.location.pathname.match(/^\/v\d+(?=\/|$)/) || [""])[0];
const maxUploadBytes = 30 * 1024 * 1024;
const maxLiteraturePdfFiles = 4;

initPageNavigation();
initSplitResizers();
updateLinkInputSummary();

literatureForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const source = event.submitter === searchAnalyzeButton ? "search" : "direct";
  await submitLiteratureAnalysisFromSource(source);
});

async function submitLiteratureAnalysisFromSource(source) {
  const statusElement = analysisStatusElement(source);
  const request = buildAnalysisRequest(source);
  if (request.error) {
    statusElement.textContent = request.error;
    if (source === "search") {
      showAppView("search", false);
      showSearchStep(request.step || 3);
    }
    return;
  }

  const runId = ++analysisRunIds[source];
  activeAnalysisSource = source;
  renderDoiPendingRows(request.previewReferences);
  renderLiteratureSummary(doiSummary, null);
  showAppView("results");
  saveAnalysisResult(source, createEmptyAnalysisResult(request.topic));
  syncActiveAnalysisResult(source);
  setLiteratureExportEnabled(false);
  setAnalysisRunning(source, true);
  statusElement.textContent = source === "search"
    ? "正在启动检索清单文献分析..."
    : "正在启动综合文献分析...";

  try {
    const response = request.pdfFiles.length
      ? await submitCombinedLiteratureAnalysis(request.references, request.pdfFiles, request.userContext, request.topic)
      : await submitLinkLiteratureAnalysis(request.references, request.userContext, request.topic, source);
    let payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    const acceptedReferences = Array.isArray(payload.references) ? payload.references : [];
    const reviewNeededDocuments = Array.isArray(payload.review_needed_documents) ? payload.review_needed_documents : [];
    if (payload.job_id) payload = await waitForJob("/api/literature-analysis", payload.job_id, (message) => {
      if (analysisRunIds[source] === runId) statusElement.textContent = message;
    });
    if (analysisRunIds[source] !== runId) return;

    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    const summary = normalizeLiteratureSummary(payload.summary);
    const result = {
      rows,
      summary,
      topic: request.topic,
      displayReferences: acceptedReferences.length ? acceptedReferences : request.previewReferences,
      reviewNeededDocuments,
    };
    saveAnalysisResult(source, result);
    if (activeAnalysisSource === source) {
      syncActiveAnalysisResult(source);
      renderDoiAnalysisRows(rows, result.displayReferences, reviewNeededDocuments);
      renderLiteratureSummary(doiSummary, summary);
      setLiteratureExportEnabled(Boolean(rows.length), Boolean(rows.length || summary));
    }
    statusElement.textContent = reviewNeededDocuments.length
      ? `已分析 ${rows.length} 篇/项文献，${reviewNeededDocuments.length} 篇待复核`
      : `已分析 ${rows.length} 篇/项文献`;
    if (source === "direct") clearSelectedPdfFiles();
  } catch (error) {
    if (analysisRunIds[source] !== runId) return;
    saveAnalysisResult(source, createEmptyAnalysisResult(request.topic));
    if (activeAnalysisSource === source) {
      syncActiveAnalysisResult(source);
      setLiteratureExportEnabled(false);
      renderDoiErrorRows(request.previewReferences, error.message);
      renderLiteratureSummary(doiSummary, null);
    }
    statusElement.textContent = source === "search" ? "检索清单文献分析失败" : "综合文献分析失败";
  } finally {
    if (analysisRunIds[source] === runId) setAnalysisRunning(source, false);
  }
}

function buildAnalysisRequest(source) {
  if (source === "search") {
    const references = stagedAnalysisReferences.map((reference) => ({ ...reference }));
    if (!references.length) {
      return { error: "请先在检索结果中加入文献到确认清单。", step: 3 };
    }
    const topic = (literatureSearchQuery.value.trim() || "literature-search-analysis").slice(0, 400);
    return {
      source,
      references,
      previewReferences: references,
      pdfFiles: [],
      userContext: "",
      topic,
    };
  }

  const entries = parseLiteratureLinkInput(doiInput.value);
  const userContext = extractLiteratureFreeText(doiInput.value);
  const pdfFiles = selectedPdfFiles;
  if (!entries.length && !pdfFiles.length && !userContext) {
    return { error: "请先提供 DOI、链接、文字说明或 PDF / DOCX" };
  }
  const linkReferences = entries.map(literatureLinkToReference);
  return {
    source,
    references: linkReferences,
    previewReferences: [...linkReferences, ...pdfFiles.map(pdfToReference)],
    pdfFiles,
    userContext,
    topic: buildLiteratureTopic(entries, pdfFiles, userContext, source),
  };
}

function createEmptyAnalysisResult(topic = "literature-analysis") {
  return {
    rows: [],
    summary: null,
    topic,
    displayReferences: [],
    reviewNeededDocuments: [],
  };
}

function saveAnalysisResult(source, result) {
  analysisResultsBySource[source] = result;
}

function syncActiveAnalysisResult(source = activeAnalysisSource) {
  const result = analysisResultsBySource[source] || createEmptyAnalysisResult();
  latestDoiAnalysisRows = result.rows;
  latestDoiAnalysisSummary = result.summary;
  latestDoiTopic = result.topic;
}

function analysisStatusElement(source) {
  return source === "search" ? literatureSearchStatus : literatureStatus;
}

pdfInput.addEventListener("change", () => {
  const files = filterSupportedUploadFiles(Array.from(pdfInput.files || []), (message) => {
    literatureStatus.textContent = message;
  });
  if (selectedPdfFiles.length + files.length > maxLiteraturePdfFiles) {
    selectedPdfFiles = [];
    pdfInput.value = "";
    updatePdfFileSummary();
    literatureStatus.textContent = `文献分析每次最多上传 ${maxLiteraturePdfFiles} 个 PDF/DOCX 文件，请重新选择。`;
    return;
  }
  addSelectedPdfFiles(files);
  pdfInput.value = "";
});

doiInput.addEventListener("input", () => {
  updateLinkInputSummary();
});

literatureSearchButton.addEventListener("click", () => {
  submitLiteratureSearch();
});

addSearchReferencesButton.addEventListener("click", () => {
  addSelectedSearchReferencesToAnalysis();
});

clearPdfButton.addEventListener("click", () => {
  clearSelectedPdfFiles();
});

literatureExportButton.addEventListener("click", () => {
  exportAnalysisDocument({
    format: literatureExportFormat.value,
    rows: latestDoiAnalysisRows,
    summary: latestDoiAnalysisSummary,
    topic: "文献分析报告",
    statusElement: literatureStatus,
  });
});

function initPageNavigation() {
  showAppView(viewFromHash(), false);
  showSearchStep(1);
  viewButtons.forEach((button) => {
    button.addEventListener("click", () => showAppView(button.dataset.viewTarget));
  });
  searchStepButtons.forEach((button) => {
    button.addEventListener("click", () => {
      showAppView("search", false);
      showSearchStep(button.dataset.searchStepTarget);
    });
  });
  window.addEventListener("hashchange", () => showAppView(viewFromHash(), false));
}

function showAppView(viewName, updateHash = true) {
  const allowed = ["home", "direct", "search", "results"];
  const nextView = allowed.includes(viewName) ? viewName : "home";
  appViews.forEach((view) => view.classList.toggle("is-active", view.dataset.appView === nextView));
  viewButtons.forEach((button) => {
    const active = button.dataset.viewTarget === nextView;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  if (updateHash && window.location.hash !== `#${nextView}`) {
    history.pushState(null, "", `#${nextView}`);
  }
}

function viewFromHash() {
  const value = window.location.hash.replace("#", "");
  const legacyMap = { literature: "home" };
  return legacyMap[value] || value || "home";
}

function showSearchStep(step) {
  const nextStep = Number(step) || 1;
  searchSteps.forEach((panel) => {
    const active = Number(panel.dataset.searchStep) === nextStep;
    panel.classList.toggle("is-active", active);
    if (active) panel.classList.remove("is-hidden");
  });
  stepperItems.forEach((item) => {
    const itemStep = Number(item.dataset.searchStepTarget);
    item.classList.toggle("is-active", itemStep === nextStep);
    item.classList.toggle("is-done", itemStep < nextStep);
  });
}

function setLiteratureExportEnabled(hasRows, hasText = hasRows) {
  const isEnabled = Boolean(hasRows || hasText);
  literatureExportFormat.disabled = !isEnabled;
  literatureExportButton.disabled = !isEnabled;
  literatureExportFormatWrap.classList.toggle("is-hidden", !isEnabled);
  literatureExportButton.classList.toggle("is-hidden", !isEnabled);
  literatureExportFormat.querySelector('option[value="md"]').disabled = !hasText;
  literatureExportFormat.querySelector('option[value="txt"]').disabled = !hasText;
  literatureExportFormat.querySelector('option[value="pdf"]').disabled = !hasText;
  if (!hasRows && hasText) literatureExportFormat.value = "txt";
  if (hasRows) literatureExportFormat.value = "md";
}

function setAnalysisRunning(source, isRunning) {
  const button = source === "search" ? searchAnalyzeButton : doiAnalyzeButton;
  const result = analysisResultsBySource[source] || createEmptyAnalysisResult();
  if (button) {
    button.disabled = isRunning;
    button.textContent = isRunning ? "分析中..." : (result.rows.length || result.summary ? "重新分析" : "开始综合分析");
  }
  if (source === "direct") {
    doiInput.disabled = isRunning;
    pdfInput.disabled = isRunning;
    clearPdfButton.disabled = isRunning || !selectedPdfFiles.length;
  }
}

function renderDoiPendingRows(references) {
  const count = Array.isArray(references) ? references.length : 0;
  renderLiteratureAnalysisLoading(
    doiAnalysisBody,
    "正在分析文献",
    count ? `已接收 ${count} 篇/项文献，LLM 正在生成贡献、方法、证据和局限分析。` : "LLM 正在生成结构化文献分析表。"
  );
}

function renderLiteratureAnalysisLoading(tbody, title, detail) {
  tbody.innerHTML = `
    <tr class="analysis-loading-row">
      <td colspan="8" class="analysis-loading-cell">
        ${loadingMarkup(title, detail)}
      </td>
    </tr>
  `;
}

function renderDoiAnalysisRows(rows, fallbackReferences, reviewNeededDocuments = []) {
  if (!rows.length && !reviewNeededDocuments.length) {
    doiAnalysisBody.innerHTML = '<tr><td colspan="8" class="empty-state">文献分析已完成，但没有返回可展示结果。</td></tr>';
    return;
  }
  const analysisRows = rows.map((row, index) => {
    const fallback = fallbackReferences[index] || {};
    return `
      <tr>
        <th>${referenceTitleCell(row, fallback)}</th>
        <td>${reviewCell(row.contribution || row.innovation)}</td>
        <td>${reviewCell(row.methodology || row.method)}</td>
        <td>${reviewCell(row.evidence_strength)}</td>
        <td>${reviewCell(innovationCellValue(row))}</td>
        <td>${reviewCell(row.limitations || row.weaknesses || row.limitation, "作者未明确说明")}</td>
        <td>${reviewCell(row.literature_positioning)}</td>
        <td>${reviewCellWithMeta(row.actionable_suggestions || row.next_step || "已完成", row.confidence)}</td>
      </tr>
    `;
  });
  const reviewRows = reviewNeededDocuments.map((document) => `
    <tr>
      <th>${referenceTitleCell(document)}</th>
      <td>待复核材料</td>
      <td>${reviewCell(document.review_note || "PDF 文本与用户预期主题或文献身份不一致，请确认上传文件或 OCR 层。")}</td>
      <td>未进入正式分析</td>
      <td>${reviewCell(toStringList(document.review_reasons).join("；") || document.pdf_identity_status || "needs_review")}</td>
      <td>需要人工确认</td>
      <td>该材料已被排除出正式文献事实抽取，避免污染分析结果。</td>
      <td>确认 PDF 后重新上传</td>
    </tr>
  `);
  doiAnalysisBody.innerHTML = [...analysisRows, ...reviewRows].join("");
}

function renderDoiErrorRows(references, message) {
  doiAnalysisBody.innerHTML = references.map((reference) => `
    <tr>
      <th>${referenceTitleCell(reference)}</th>
      <td>分析失败</td><td>未生成</td><td>未生成</td><td>未生成</td><td>未生成</td>
      <td>${escapeHtml(message)}</td><td>失败</td>
    </tr>
  `).join("");
}

function renderLiteratureSummary(container, summary) {
  if (!container) return;
  const normalized = normalizeLiteratureSummary(summary);
  if (!normalized) {
    container.classList.add("is-empty");
    container.classList.remove("is-resizable");
    updateSplitResizerVisibility(container);
    container.innerHTML = "";
    return;
  }
  const groups = [
    ["共同优势", normalized.common_strengths],
    ["共同弱点", normalized.common_weaknesses],
    ["方法模式", normalized.methodological_patterns],
    ["证据缺口", normalized.evidence_gaps],
    ["研究空白", normalized.research_gaps],
    ["推荐阅读顺序", normalized.recommended_reading_order],
    ["参考文献", normalized.references],
    ["事实风险提示", normalized.fact_risks],
    ["后续行动", normalized.next_actions],
  ].filter(([, items]) => items.length);
  container.classList.remove("is-empty");
  container.classList.add("is-resizable");
  container.innerHTML = `
    <h3>跨文献总结</h3>
    <p class="summary-lead">${escapeHtml(normalized.overall_assessment || "已生成跨文献总结。")}</p>
    <div class="summary-grid">
      ${groups.map(([title, items]) => summaryGroup(title, items)).join("")}
      ${normalized.confidence ? summaryGroup("置信度", [normalized.confidence]) : ""}
    </div>
  `;
  updateSplitResizerVisibility(container);
}

function initSplitResizers() {
  document.querySelectorAll(".split-resizer[data-resize-target]").forEach((handle) => {
    const target = document.getElementById(handle.dataset.resizeTarget);
    if (!target) return;
    handle.addEventListener("pointerdown", (event) => startSplitResize(event, handle, target));
    handle.addEventListener("keydown", (event) => {
      if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const current = Number.parseInt(target.style.getPropertyValue("--summary-height"), 10) ||
        Math.round(target.getBoundingClientRect().height);
      let next = current;
      if (event.key === "ArrowUp") next -= 32;
      if (event.key === "ArrowDown") next += 32;
      if (event.key === "Home") next = 140;
      if (event.key === "End") next = 640;
      setSummaryHeight(target, next);
    });
    updateSplitResizerVisibility(target);
  });
}

function innovationCellValue(row) {
  return row.what_is_new || row.innovation_point || row.innovation || row.strengths || "";
}

function normalizeDateInput(value, boundary) {
  const cleaned = String(value || "").trim();
  if (!cleaned) return "";
  const yearOnly = cleaned.match(/^(\d{4})$/);
  if (yearOnly) return boundary === "end" ? `${yearOnly[1]}-12-31` : `${yearOnly[1]}-01-01`;
  const yearMonth = cleaned.match(/^(\d{4})[-/.](\d{1,2})$/);
  if (yearMonth) {
    const year = Number(yearMonth[1]);
    const month = Number(yearMonth[2]);
    if (month < 1 || month > 12) return "";
    if (boundary === "end") {
      return toDateString(new Date(year, month, 0));
    }
    return `${yearMonth[1]}-${String(month).padStart(2, "0")}-01`;
  }
  const full = cleaned.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$/);
  if (!full) return cleaned;
  return `${full[1]}-${String(Number(full[2])).padStart(2, "0")}-${String(Number(full[3])).padStart(2, "0")}`;
}

function toDateString(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function startSplitResize(event, handle, target) {
  if (event.button !== 0) return;
  event.preventDefault();
  const startY = event.clientY;
  const startHeight = target.getBoundingClientRect().height;
  document.body.classList.add("is-resizing-split");
  handle.setPointerCapture(event.pointerId);

  const onMove = (moveEvent) => {
    const nextHeight = startHeight + moveEvent.clientY - startY;
    setSummaryHeight(target, nextHeight);
  };
  const onEnd = () => {
    document.body.classList.remove("is-resizing-split");
    handle.removeEventListener("pointermove", onMove);
    handle.removeEventListener("pointerup", onEnd);
    handle.removeEventListener("pointercancel", onEnd);
  };

  handle.addEventListener("pointermove", onMove);
  handle.addEventListener("pointerup", onEnd);
  handle.addEventListener("pointercancel", onEnd);
}

function setSummaryHeight(target, value) {
  const height = Math.max(140, Math.min(720, Math.round(value)));
  target.style.setProperty("--summary-height", `${height}px`);
}

function updateSplitResizerVisibility(target) {
  const handle = document.querySelector(`.split-resizer[data-resize-target="${target.id}"]`);
  if (!handle) return;
  handle.classList.toggle("is-visible", !target.classList.contains("is-empty"));
}

function normalizeLiteratureSummary(summary) {
  if (!summary || typeof summary !== "object") return null;
  const normalized = {
    overall_assessment: String(summary.overall_assessment || "").trim(),
    common_strengths: toStringList(summary.common_strengths),
    common_weaknesses: toStringList(summary.common_weaknesses),
    methodological_patterns: toStringList(summary.methodological_patterns),
    evidence_gaps: toStringList(summary.evidence_gaps),
    research_gaps: toStringList(summary.research_gaps),
    recommended_reading_order: toStringList(summary.recommended_reading_order),
    references: toStringList(summary.references),
    next_actions: toStringList(summary.next_actions),
    fact_risks: toStringList(summary.fact_risks),
    confidence: String(summary.confidence || "").trim(),
  };
  const hasContent = normalized.overall_assessment || normalized.confidence ||
    Object.values(normalized).some((value) => Array.isArray(value) && value.length);
  return hasContent ? normalized : null;
}

function summaryGroup(title, items) {
  return `<div class="summary-group"><strong>${escapeHtml(title)}</strong><ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>`;
}

function toStringList(value) {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

function parseLiteratureLinkInput(value) {
  const doiMatches = value.match(/10\.\d{4,9}\/[^\s,;，；]+/gi) || [];
  const urlMatches = value.match(/https?:\/\/[^\s,;，；]+/gi) || [];
  const pmidMatches = value.match(/\bPMID\s*:?\s*\d{6,9}\b/gi) || [];
  const barePmidMatches = String(value || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^\d{6,9}$/.test(line));
  const seen = new Set();
  const entries = [];
  doiMatches.forEach((doi) => addLiteratureEntry(entries, seen, { type: "doi", value: doi.replace(/[.)\]}]+$/g, "") }));
  urlMatches.forEach((url) => {
    const cleaned = url.replace(/[.)\]}]+$/g, "");
    const doi = extractDoiFromText(cleaned);
    const pmid = extractPmidFromText(cleaned);
    addLiteratureEntry(entries, seen, doi ? { type: "doi", value: doi } : (pmid ? { type: "pmid", value: pmid } : { type: "url", value: cleaned }));
  });
  pmidMatches.forEach((pmid) => {
    addLiteratureEntry(entries, seen, { type: "pmid", value: extractPmidFromText(pmid) });
  });
  barePmidMatches.forEach((pmid) => {
    addLiteratureEntry(entries, seen, { type: "pmid", value: pmid });
  });
  return entries;
}

function addLiteratureEntry(entries, seen, entry) {
  const key = `${entry.type}:${entry.value.toLowerCase()}`;
  if (seen.has(key)) return;
  seen.add(key);
  entries.push(entry);
}

function extractLiteratureFreeText(value) {
  const doiMatches = value.match(/10\.\d{4,9}\/[^\s,;，；]+/gi) || [];
  const urlMatches = value.match(/https?:\/\/[^\s,;，；]+/gi) || [];
  let text = String(value || "");
  [...doiMatches, ...urlMatches].forEach((token) => {
    text = text.replace(token, " ");
  });
  return text
    .split(/\n{2,}/)
    .map((block) => block.replace(/\s+/g, " ").trim())
    .filter((block) => block.length >= 8)
    .join("\n\n");
}

function extractDoiFromText(value) {
  const match = value.match(/10\.\d{4,9}\/[^\s,;，；]+/i);
  return match ? match[0].replace(/[.)\]}]+$/g, "") : "";
}

function extractPmidFromText(value) {
  const pubmedUrlMatch = value.match(/pubmed\.ncbi\.nlm\.nih\.gov\/(\d{6,9})/i);
  if (pubmedUrlMatch) return pubmedUrlMatch[1];
  const legacyUrlMatch = value.match(/ncbi\.nlm\.nih\.gov\/pubmed\/(\d{6,9})/i);
  if (legacyUrlMatch) return legacyUrlMatch[1];
  const pmidMatch = value.match(/\bPMID\s*:?\s*(\d{6,9})\b/i);
  return pmidMatch ? pmidMatch[1] : "";
}

function literatureLinkToReference(entry) {
  if (entry.type === "doi") {
    return {
      title: `DOI: ${entry.value}`,
      source: `https://doi.org/${entry.value}`,
      relevance: "用户在文献分析中主动提交，需要进行文献分析。",
      branch_name: "文献分析",
    };
  }
  if (entry.type === "pmid") {
    return {
      title: `PMID: ${entry.value}`,
      source: `https://pubmed.ncbi.nlm.nih.gov/${entry.value}/`,
      relevance: "用户在文献分析中主动提交 PubMed 文献，需要进行文献分析。",
      branch_name: "文献分析",
    };
  }
  return {
    title: readableTitleFromUrl(entry.value),
    source: entry.value,
    relevance: "用户在文献分析中主动提交论文链接，需要进行文献分析。",
    branch_name: "文献分析",
  };
}

function pdfToReference(file) {
  return {
    title: file.name.replace(/\.(pdf|docx)$/i, ""),
    source: file.name,
    relevance: "用户上传文件，需要进行文献分析。",
    branch_name: "文件上传",
  };
}

function isResearchDocument(file) {
  return /\.pdf$/i.test(file.name) ||
    /\.docx$/i.test(file.name) ||
    file.type === "application/pdf" ||
    file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
}

function filterSupportedUploadFiles(files, report) {
  const legacyWordFiles = files.filter((file) => /\.doc$/i.test(file.name));
  if (legacyWordFiles.length) {
    report(`不支持旧版 .doc 文件：${legacyWordFiles.map((file) => file.name).join("、")}。请另存为 .docx 或 PDF 后再上传。`);
  }
  const supported = files.filter(isResearchDocument);
  const unsupported = files.filter((file) => !isResearchDocument(file) && !/\.doc$/i.test(file.name));
  if (unsupported.length) {
    report(`不支持这些文件：${unsupported.map((file) => file.name).join("、")}。目前只支持 PDF / DOCX。`);
  }
  return supported;
}

function addSelectedPdfFiles(files) {
  if (!files.length) {
    updatePdfFileSummary();
    return;
  }
  const existing = new Set(selectedPdfFiles.map(pdfFileKey));
  files.forEach((file) => {
    const key = pdfFileKey(file);
    if (!existing.has(key)) {
      selectedPdfFiles.push(file);
      existing.add(key);
    }
  });
  updatePdfFileSummary();
}

function clearSelectedPdfFiles() {
  selectedPdfFiles = [];
  pdfInput.value = "";
  updatePdfFileSummary();
}

function updatePdfFileSummary() {
  if (!selectedPdfFiles.length) {
    pdfFileSummary.textContent = "尚未选择文件。";
    clearPdfButton.disabled = true;
    return;
  }
  const totalSize = selectedPdfFiles.reduce((sum, file) => sum + file.size, 0);
  pdfFileSummary.textContent = formatSelectedFilesSummary(selectedPdfFiles);
  if (totalSize > maxUploadBytes) {
    pdfFileSummary.textContent += `，已超过 ${formatFileSize(maxUploadBytes)} 上传上限`;
  }
  clearPdfButton.disabled = false;
}

function updateLinkInputSummary() {
  if (!linkInputSummary || !doiInput) return;
  const value = doiInput.value || "";
  const entries = parseLiteratureLinkInput(value);
  const userContext = extractLiteratureFreeText(value);
  if (!entries.length && !userContext) {
    linkInputSummary.textContent = "尚未添加链接或文字说明。";
    return;
  }
  const counts = entries.reduce((acc, entry) => {
    acc[entry.type] = (acc[entry.type] || 0) + 1;
    return acc;
  }, {});
  const parts = [
    counts.doi ? `${counts.doi} 个 DOI` : "",
    counts.pmid ? `${counts.pmid} 个 PMID` : "",
    counts.url ? `${counts.url} 个链接` : "",
    userContext ? "含文字说明" : "",
  ].filter(Boolean);
  linkInputSummary.textContent = `已识别：${parts.join("，")}`;
}

function pdfFileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function readableTitleFromUrl(url) {
  try {
    const parsed = new URL(url);
    const part = decodeURIComponent(parsed.pathname.split("/").filter(Boolean).pop() || parsed.hostname);
    return part.replace(/[-_]+/g, " ") || parsed.hostname;
  } catch (error) {
    return url;
  }
}

async function submitLiteratureSearch() {
  const query = literatureSearchQuery.value.trim();
  if (!query) {
    literatureSearchStatus.textContent = "请输入研究主题或检索式。";
    showAppView("search", false);
    showSearchStep(1);
    return;
  }
  const sources = $$('input[name="paperSource"]:checked').map((input) => input.value).join(",");
  if (!sources) {
    literatureSearchStatus.textContent = "请至少选择一个检索来源。";
    showAppView("search", false);
    showSearchStep(1);
    return;
  }
  literatureSearchButton.disabled = true;
  literatureSearchStatus.textContent = "正在检索候选文献...";
  searchCandidatePanel.classList.remove("is-hidden");
  showAppView("search", false);
  showSearchStep(2);
  searchCandidateList.innerHTML = loadingMarkup("正在检索", "正在调用可选 paper-search 集成，并进行本地筛选与校验。");
  try {
    const response = await fetch(apiPath("/api/literature-search"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        sources,
        max_results_per_source: Number(literatureSearchLimit.value || 5),
        year: literatureSearchYear.value.trim(),
        include_needs_review: includeNeedsReviewSearch.checked,
        append_annotation_record: appendAnnotationRecordSearch ? appendAnnotationRecordSearch.checked : true,
      }),
    });
    const payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    const qualified = (payload.qualified_references || []).map((reference) => ({
      ...reference,
      candidate_group: "qualified",
    }));
    const needsReview = (payload.needs_review_references || []).map((reference) => ({
      ...reference,
      candidate_group: "needs_review",
    }));
    searchCandidateReferences = [...qualified, ...needsReview].map((reference, index) => ({
      ...reference,
      candidate_id: reference.dedupe_key || reference.source || reference.doi || reference.title || `candidate-${index}`,
    }));
    selectedSearchReferenceIds = new Set(qualified.map((reference, index) => (
      reference.dedupe_key || reference.source || reference.doi || reference.title || `candidate-${index}`
    )));
    renderSearchCandidates(payload);
    literatureSearchStatus.textContent = `检索完成：合格 ${qualified.length} 条，待复核 ${needsReview.length} 条，已过滤 ${payload.rejected_count || 0} 条。`;
    showSearchStep(2);
  } catch (error) {
    searchCandidateReferences = [];
    selectedSearchReferenceIds = new Set();
    renderSearchCandidates({ rejected_count: 0, errors: { search: error.message } });
    literatureSearchStatus.textContent = `检索失败：${error.message}`;
    showSearchStep(2);
  } finally {
    literatureSearchButton.disabled = false;
  }
}

function renderSearchCandidates(payload = {}) {
  const rejectedCount = Number(payload.rejected_count || 0);
  const errors = payload.errors || {};
  const errorText = Object.entries(errors).map(([source, message]) => `${source}: ${message}`).join("；");
  searchCandidateMeta.textContent = [
    `候选 ${searchCandidateReferences.length} 条`,
    rejectedCount ? `已过滤 ${rejectedCount} 条` : "",
    errorText ? `部分来源提示：${errorText}` : "",
  ].filter(Boolean).join(" · ");
  addSearchReferencesButton.disabled = selectedSearchReferenceIds.size === 0;
  if (!searchCandidateReferences.length) {
    searchCandidateList.innerHTML = '<div class="empty-state candidate-empty">暂无可展示候选。请调整检索式、来源或年份范围。</div>';
    renderStagedAnalysisReferences();
    return;
  }
  const qualified = searchCandidateReferences.filter((reference) => reference.candidate_group !== "needs_review");
  const needsReview = searchCandidateReferences.filter((reference) => reference.candidate_group === "needs_review");
  searchCandidateList.innerHTML = [
    qualified.length ? `<section class="candidate-group"><h3>合格候选</h3>${candidateItemsMarkup(qualified)}</section>` : "",
    needsReview.length ? `<section class="candidate-group"><h3>待复核候选</h3>${candidateItemsMarkup(needsReview)}</section>` : "",
  ].filter(Boolean).join("");
  $$("[data-candidate-id]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => toggleSearchCandidate(checkbox.dataset.candidateId, checkbox.checked));
  });
  renderStagedAnalysisReferences();
}

function candidateItemsMarkup(references) {
  return references.map((reference) => {
    const id = reference.candidate_id;
    const isChecked = selectedSearchReferenceIds.has(id);
    const status = reference.candidate_group === "needs_review" ? "needs_review" : reference.screening_status;
    const risks = [...toStringList(reference.screening_risks), ...toStringList(reference.verification_risks)];
    return `
      <article class="candidate-item ${reference.candidate_group === "needs_review" ? "needs-review" : ""}">
        <label class="candidate-check">
          <input type="checkbox" data-candidate-id="${escapeHtml(id)}" ${isChecked ? "checked" : ""} />
          <span>${referenceVerificationBadge(reference)} ${referenceScreeningBadge(status)}</span>
        </label>
        <div class="candidate-main">
          <h3>${linkOrText(reference.title || "未命名文献", reference.source || "")}</h3>
          <p class="candidate-byline">${escapeHtml([reference.authors, reference.year, reference.source_label].filter(Boolean).join(" · ") || "元数据不完整")}</p>
          <p class="candidate-idline">${escapeHtml(referenceIdentifierText(reference))}</p>
          <p class="candidate-abstract">${escapeHtml(reference.abstract || reference.relevance || "暂无摘要。")}</p>
          ${risks.length ? `<p class="candidate-risks">风险：${escapeHtml(risks.join("；"))}</p>` : ""}
        </div>
      </article>
    `;
  }).join("");
}

function toggleSearchCandidate(candidateId, isSelected) {
  if (isSelected) {
    selectedSearchReferenceIds.add(candidateId);
  } else {
    selectedSearchReferenceIds.delete(candidateId);
  }
  addSearchReferencesButton.disabled = selectedSearchReferenceIds.size === 0;
}

function addSelectedSearchReferencesToAnalysis() {
  const existing = new Set(stagedAnalysisReferences.map(referenceStableKey));
  const selected = searchCandidateReferences
    .filter((reference) => selectedSearchReferenceIds.has(reference.candidate_id))
    .map(searchCandidateToAnalysisReference)
    .filter((reference) => {
      const key = referenceStableKey(reference);
      if (existing.has(key)) return false;
      existing.add(key);
      return true;
    });
  stagedAnalysisReferences = [...stagedAnalysisReferences, ...selected];
  renderStagedAnalysisReferences();
  literatureStatus.textContent = `已加入 ${selected.length} 篇检索文献到确认清单。点击“开始综合分析”后才会进入 workflow。`;
  literatureSearchStatus.textContent = `已加入 ${selected.length} 篇文献到确认清单。`;
  showAppView("search", false);
  showSearchStep(3);
}

function renderStagedAnalysisReferences() {
  if (!stagedAnalysisReferences.length) {
    stagedReferenceList.classList.add("empty-selection");
    stagedReferenceList.innerHTML = "<strong>尚未加入文献</strong><p>请返回第 2 步，在检索结果中勾选文献并点击添加。</p>";
    return;
  }
  stagedReferenceList.classList.remove("empty-selection");
  stagedReferenceList.innerHTML = `
    <strong>已加入分析列表：${stagedAnalysisReferences.length} 篇</strong>
    <div class="selected-reference-list">
      ${stagedAnalysisReferences.map((reference, index) => `
        <article class="selected-reference-item">
          <div>
            <span class="selected-reference-index">${index + 1}</span>
            <strong>${escapeHtml(reference.title || "未命名文献")}</strong>
            <small>${escapeHtml(referenceIdentifierText(reference))} · ${escapeHtml(reference.verification_status || "partial")}</small>
          </div>
          <button class="ghost-button selected-reference-remove" type="button" data-remove-staged-reference="${index}">移除</button>
        </article>
      `).join("")}
    </div>
  `;
  $$("[data-remove-staged-reference]").forEach((button) => {
    button.addEventListener("click", () => removeStagedAnalysisReference(Number(button.dataset.removeStagedReference)));
  });
}

function removeStagedAnalysisReference(index) {
  if (index < 0 || index >= stagedAnalysisReferences.length) return;
  const [removed] = stagedAnalysisReferences.splice(index, 1);
  if (removed) {
    const removedKey = referenceStableKey(removed);
    searchCandidateReferences.forEach((candidate) => {
      if (referenceStableKey(candidate) === removedKey) {
        selectedSearchReferenceIds.delete(candidate.candidate_id);
      }
    });
    syncSearchCandidateSelectionControls();
  }
  renderStagedAnalysisReferences();
  literatureSearchStatus.textContent = stagedAnalysisReferences.length
    ? `确认清单剩余 ${stagedAnalysisReferences.length} 篇文献。`
    : "确认清单已清空。";
}

function syncSearchCandidateSelectionControls() {
  $$("input[data-candidate-id]").forEach((checkbox) => {
    checkbox.checked = selectedSearchReferenceIds.has(checkbox.dataset.candidateId);
  });
  addSearchReferencesButton.disabled = selectedSearchReferenceIds.size === 0;
}

function searchCandidateToAnalysisReference(reference) {
  const {
    raw_source_record,
    candidate_id,
    candidate_group,
    ...rest
  } = reference;
  return {
    ...rest,
    source_origin: rest.source_origin || "paper_search_mcp",
    document_role: "literature",
    is_literature_source: true,
  };
}

function referenceStableKey(reference) {
  return String(reference.dedupe_key || reference.doi || reference.pmid || reference.arxiv_id || reference.source || reference.title || "").toLowerCase();
}

function referenceVerificationBadge(reference) {
  const status = reference.verification_status || "partial";
  return `<span class="candidate-badge verification-${escapeHtml(status)}">${escapeHtml(status)}</span>`;
}

function referenceScreeningBadge(status) {
  const value = status || "qualified";
  return `<span class="candidate-badge screening-${escapeHtml(value)}">${escapeHtml(value)}</span>`;
}

function referenceIdentifierText(reference) {
  if (reference.doi) return `DOI: ${reference.doi}`;
  if (reference.arxiv_id) return `arXiv: ${reference.arxiv_id}`;
  if (reference.pmid) return `PMID: ${reference.pmid}`;
  return reference.source || "无稳定 ID";
}

async function submitLiteratureAnalysis({ topic = "literature-analysis", references = [], finalReport = "" } = {}) {
  return fetch(apiPath("/api/literature-analysis"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      references,
      final_report: finalReport,
    }),
  });
}

async function submitLinkLiteratureAnalysis(references, userContext = "", topic = "literature-analysis", analysisSource = "direct") {
  const fallback = analysisSource === "search"
    ? "The user selected these references from the literature search flow for final literature analysis."
    : "The user provided DOI identifiers or literature links directly in the literature assistant.";
  return submitLiteratureAnalysis({
    topic,
    references,
    finalReport: buildLiteratureUserContext(userContext, fallback),
  });
}

async function submitCombinedLiteratureAnalysis(references, pdfFiles, userContext = "", topic = "literature-analysis") {
  assertUploadSize(pdfFiles);
  const formData = new FormData();
  formData.append("topic", topic);
  formData.append("references", JSON.stringify(references));
  formData.append("user_context", userContext);
  pdfFiles.forEach((file) => formData.append("pdf", file));
  return fetch(apiPath("/api/literature-analysis/pdf"), { method: "POST", body: formData });
}

function buildLiteratureTopic(entries, pdfFiles, userContext = "", source = "direct") {
  const context = String(userContext || "").trim();
  if (context) return context.slice(0, 400);
  if (source === "search" && literatureSearchQuery.value.trim()) return literatureSearchQuery.value.trim().slice(0, 400);
  if (entries.length) return entries.map((entry) => entry.value).slice(0, 3).join(" ");
  if (pdfFiles.length) return pdfFiles.map((file) => file.name).slice(0, 3).join(" ");
  return "literature-analysis";
}

function buildLiteratureUserContext(userContext, fallback) {
  const context = String(userContext || "").trim();
  if (!context) return fallback;
  return `${fallback}\n\nUser-provided text context or instructions:\n${context}`;
}

async function waitForJob(basePath, jobId, updateStatus) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 10 * 60 * 1000) {
    await sleep(1000);
    const response = await fetch(apiPath(`${basePath}/${jobId}`), { cache: "no-store" });
    const payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.status === "done") return payload;
    if (payload.status === "error") throw new Error(payload.error || "任务运行失败。");
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    updateStatus(translateJobStage(payload.stage) || `运行中，已运行 ${elapsed} 秒...`);
  }
  throw new Error("任务超过 10 分钟未完成，已停止等待。");
}

function translateJobStage(stage) {
  const text = String(stage || "").trim();
  const stageMap = {    "Starting literature analysis...": "正在启动文献分析...",
    "Resolving DOI metadata...": "正在补全文献元数据...",
    "Running LLM literature analysis...": "正在运行文献分析...",
  };
  return stageMap[text] || text;
}

async function readJsonResponse(response) {
  const responseText = await response.text();
  if (!responseText) return {};
  try {
    return JSON.parse(responseText);
  } catch (error) {
    throw new Error(`服务返回了无法解析的数据：${responseText.slice(0, 160)}`);
  }
}

function loadingMarkup(title, message) {
  return `
    <div class="loading-state" role="status" aria-live="polite">
      <div class="loading-squares" aria-hidden="true">
        <span class="loading-square loading-square-large"></span>
        <span class="loading-square loading-square-medium"></span>
        <span class="loading-square loading-square-small"></span>
      </div>
      <div class="loading-copy">
        <h2>${escapeHtml(title)}</h2>
        <p>${escapeHtml(message)}</p>
      </div>
    </div>
  `;
}

function reviewCell(value, unclearLabel = "未明确说明") {
  return escapeHtml(publicCellText(value, unclearLabel));
}

function reviewCellWithMeta(value, meta) {
  const body = reviewCell(value);
  return meta ? `${body}<br><small>${escapeHtml(meta)}</small>` : body;
}

function referenceTitleCell(row, fallback = {}) {
  const title = row.title || fallback.title || "未命名文献";
  const source = row.source || fallback.source || "";
  const uploadedFilename = row.uploaded_filename || fallback.uploaded_filename || "";
  return linkOrText(title, source, uploadedFilename);
}

function linkOrText(title, source, uploadedFilename = "") {
  const titleText = String(title || "未命名").trim();
  const sourceText = String(source || "").trim();
  const fileText = String(uploadedFilename || "").trim();
  const cleanTitle = escapeHtml(titleText);
  const displaySource = fileText || sourceText;
  const sourceLine = displaySource && displaySource.toLowerCase() !== titleText.toLowerCase()
    ? `<small class="source-filename">${escapeHtml(sourceLabel(displaySource, Boolean(fileText)))}</small>`
    : "";
  if (/^https?:\/\//i.test(sourceText)) {
    return `<a href="${escapeHtml(sourceText)}" target="_blank" rel="noreferrer">${cleanTitle}</a>${sourceLine}`;
  }
  return `${cleanTitle}${sourceLine}`;
}

function sourceLabel(source, forceFile = false) {
  return forceFile || !/^https?:\/\//i.test(source || "") ? `文件：${source}` : `来源：${source}`;
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function downloadPdfDocument({ title, markdown, filename, onStatus }) {
  onStatus("正在生成 PDF...");
  try {
    const response = await fetch(apiPath("/api/export/pdf"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, markdown }),
    });
    if (!response.ok) {
      const payload = await readJsonResponse(response);
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    downloadBlob(filename, blob);
    onStatus("PDF 已下载。");
  } catch (error) {
    onStatus(`PDF 导出失败：${error.message}`, true);
  }
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function exportAnalysisDocument({ format, rows, summary, topic, statusElement }) {
  if (!rows.length && !summary) return;
  const markdown = buildAnalysisMarkdown(rows, summary);
  const reportTitle = analysisReportTitle(topic);
  const baseName = safeFileName(reportTitle);
  if (format === "pdf") {
    downloadPdfDocument({
      title: reportTitle,
      markdown,
      filename: `${baseName}.pdf`,
      onStatus: (message, isError = false) => {
        statusElement.textContent = message;
        statusElement.classList.toggle("error", isError);
      },
    });
    return;
  }
  const extension = format === "txt" ? "txt" : "md";
  const label = extension.toUpperCase();
  downloadText(`${baseName}.${extension}`, markdown);
  statusElement.textContent = `文献分析 ${label} 已下载。`;
}

function analysisReportTitle(topic) {
  const title = String(topic || "").trim() || "文献分析报告";
  return /文献分析|literature analysis/i.test(title) ? title : `${title}_文献分析`;
}

function buildAnalysisMarkdown(rows, summary) {
  const summaryText = summaryToText(summary);
  const rowsText = rows.map((row, index) => {
    const columns = analysisExportColumns();
    const uploadedFilename = String(row.uploaded_filename || "").trim();
    const source = String(row.source || "").trim();
    return [
      `## ${index + 1}. ${row.title || "未命名文献"}`,
      uploadedFilename ? `文件：${uploadedFilename}` : "",
      source && source !== uploadedFilename ? `来源：${source}` : "",
      referenceProvenanceText(row),
      buildFactSlotTable(row),
      "### 评审摘要",
      ...columns.slice(1).map((column) => `**${column.label}：** ${exportCellValue(column.value(row), column.fallback || "未说明")}`),
    ].filter(Boolean).join("\n\n");
  }).join("\n\n");
  return [summaryText, rowsText].filter(Boolean).join("\n\n---\n\n") || "暂无可导出的文献分析内容。";
}

function summaryToText(summary) {
  if (!summary) return "";
  const sections = [];
  if (summary.overall_assessment) sections.push(`# 跨文献总结\n\n${summary.overall_assessment}`);
  [
    ["共同优势", summary.common_strengths],
    ["共同弱点", summary.common_weaknesses],
    ["方法模式", summary.methodological_patterns],
    ["证据缺口", summary.evidence_gaps],
    ["研究空白", summary.research_gaps],
    ["推荐阅读顺序", summary.recommended_reading_order],
    ["事实风险提示", summary.fact_risks],
    ["后续行动", summary.next_actions],
    ["参考文献", summary.references],
  ].forEach(([title, items]) => {
    const list = toStringList(items);
    if (list.length) {
      sections.push(`## ${title}\n\n${list.map((item) => `- ${item}`).join("\n")}`);
    }
  });
  if (summary.confidence) sections.push(`## 置信度\n\n${summary.confidence}`);
  return sections.join("\n\n");
}

function referenceProvenanceText(row) {
  const provenance = row.provenance && typeof row.provenance === "object" ? row.provenance : {};
  const searchSource = [row.source_origin, row.source_label || provenance.retrieved_from].filter(Boolean).join(" / ");
  const evidenceLevel = provenance.evidence_level || (row.pdf_text_available ? "full_text" : (row.abstract ? "metadata+abstract" : "metadata"));
  const risks = [...toStringList(row.verification_risks), ...toStringList(row.screening_risks)];
  const lines = [
    searchSource ? `检索来源：${searchSource}` : "",
    row.topic_relevance_status ? `主题相关性：${row.topic_relevance_status}${row.topic_relevance_score ? ` (${row.topic_relevance_score})` : ""}` : "",
    row.verification_status ? `校验状态：${row.verification_status}` : "",
    evidenceLevel ? `证据级别：${evidenceLevel}` : "",
    `风险提示：${risks.length ? risks.join("；") : "无"}`,
  ].filter(Boolean);
  return lines.length ? lines.join("\n\n") : "";
}

function buildFactSlotTable(row) {
  const factRows = factSlotColumns().map((column) => [
    markdownTableCell(column.label),
    markdownTableCell(exportCellValue(column.value(row), column.fallback || "未明确说明")),
  ]);
  return [
    "### 通用事实槽",
    "| 字段 | 抽取结果 |",
    "| --- | --- |",
    ...factRows.map(([label, value]) => `| ${label} | ${value} |`),
  ].join("\n");
}

function factSlotColumns() {
  return [
    { label: "study_type / 研究类型", value: (row) => row.study_type || row.paper_type, fallback: "未明确说明研究类型" },
    { label: "research_objective / 研究目标", value: (row) => row.research_objective || row.task || row.core_claim, fallback: "未明确说明研究目标" },
    { label: "dataset_or_material / 数据集、材料、对象", value: (row) => row.dataset_or_material || row.dataset, fallback: "未明确说明数据/材料" },
    { label: "sample_size / 样本量、实验规模、数据量", value: (row) => row.sample_size, fallback: "未明确报告样本量/规模" },
    { label: "domain_or_modality / 领域、模态、对象类型", value: (row) => row.domain_or_modality || row.modality, fallback: "未明确说明领域/模态" },
    { label: "method / 方法、模型、干预、框架", value: (row) => row.method || row.model_or_method || row.methodology, fallback: "未明确说明方法" },
    { label: "baseline_or_comparator / 对照、基线、比较对象", value: (row) => row.baseline_or_comparator || row.baseline, fallback: "未明确说明对照/基线" },
    { label: "evaluation_protocol / 评价或验证方案", value: (row) => row.evaluation_protocol || row.validation_setup, fallback: "未明确说明评价/验证方案" },
    { label: "metrics / 评价指标", value: (row) => row.metrics, fallback: "未明确报告评价指标" },
    { label: "key_results / 关键结果", value: (row) => row.key_results, fallback: "未明确报告关键结果" },
    { label: "statistical_evidence / 统计证据", value: (row) => row.statistical_evidence, fallback: "未明确报告统计证据" },
    { label: "availability / 代码、数据、模型、材料公开性", value: (row) => row.availability, fallback: "未明确说明公开性" },
    { label: "limitations / 作者承认的局限", value: (row) => row.limitations, fallback: "作者未明确说明" },
    { label: "evidence_locations / 证据位置", value: (row) => row.evidence_locations, fallback: "未定位到证据位置" },
  ];
}

function debugFactSlotColumns() {
  return [
    ...factSlotColumns(),
    { label: "fact_risks / 单篇事实风险", value: (row) => row.fact_risks },
    { label: "evidence_candidates / 候选证据", value: (row) => row.evidence_candidates },
  ];
}

function exportCellValue(value, fallback = "unclear") {
  const items = toStringList(value);
  if (items.length) return items.map((item) => publicCellText(item, fallback)).join("; ");
  const text = String(value == null ? "" : value).trim();
  return publicCellText(text || fallback, fallback);
}

function publicCellText(value, unclearLabel = "未明确说明") {
  const text = String(value == null ? "" : value).trim();
  if (!text || /^unclear$/i.test(text)) return unclearLabel;
  return text.replace(/\bunclear\b/gi, unclearLabel);
}

function markdownTableCell(value) {
  return exportCellValue(value, "unclear")
    .replace(/\r?\n+/g, "<br>")
    .replace(/\|/g, "\\|");
}

function analysisExportColumns() {
  return [
    { label: "文献/来源", value: (row) => row.title || "" },
    { label: "链接", value: (row) => row.source || "" },
    { label: "检索来源", value: (row) => [row.source_origin, row.source_label].filter(Boolean).join(" / ") || "" },
    { label: "校验状态", value: (row) => row.verification_status || "" },
    { label: "主题相关性", value: (row) => row.topic_relevance_status ? `${row.topic_relevance_status}${row.topic_relevance_score ? ` (${row.topic_relevance_score})` : ""}` : "" },
    { label: "证据级别", value: (row) => row.provenance && row.provenance.evidence_level || "" },
    { label: "风险提示", value: (row) => [...toStringList(row.verification_risks), ...toStringList(row.screening_risks)].join("；") || "无" },
    { label: "核心贡献", value: (row) => row.contribution || row.innovation || "" },
    { label: "方法/证据", value: (row) => row.methodology || row.method || "" },
    { label: "证据强度", value: (row) => row.evidence_strength || "" },
    { label: "创新点", value: (row) => row.innovation_point || row.innovation || row.strengths || "" },
    { label: "主要局限", value: (row) => row.limitations || row.weaknesses || row.limitation || "", fallback: "作者未明确说明" },
    { label: "文献定位", value: (row) => row.literature_positioning || "" },
    { label: "后续建议", value: (row) => row.actionable_suggestions || row.next_step || "" },
    { label: "置信度", value: (row) => row.confidence || "" },
  ];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatSelectedFilesSummary(files) {
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  const names = files.map((file) => file.name).slice(0, 3).join("、");
  const suffix = files.length > 3 ? `，另有 ${files.length - 3} 个文件` : "";
  return `已添加 ${files.length} 个文件，约 ${formatFileSize(totalSize)}：${names}${suffix}`;
}

function assertUploadSize(files) {
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  if (totalSize > maxUploadBytes) {
    throw new Error(`上传文件总大小约 ${formatFileSize(totalSize)}，超过 ${formatFileSize(maxUploadBytes)} 上限。请减少文件数量，少量多次上传。`);
  }
}

function apiPath(path) {
  return `${appBasePath}${path}`;
}

function safeFileName(value) {
  return (value || "literature-analysis")
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80) || "literature-analysis";
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}



