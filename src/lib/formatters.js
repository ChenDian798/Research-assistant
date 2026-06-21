export const maxUploadBytes = 30 * 1024 * 1024;
export const maxLiteraturePdfFiles = 4;

export function createEmptyAnalysisResult(topic = "literature-analysis") {
  return {
    rows: [],
    summary: null,
    topic,
    displayReferences: [],
    reviewNeededDocuments: [],
  };
}

export function normalizeLiteratureSummary(summary) {
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

export function toStringList(value) {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

export function parseLiteratureLinkInput(value) {
  const doiMatches = String(value || "").match(/10\.\d{4,9}\/[^\s,;，；]+/gi) || [];
  const urlMatches = String(value || "").match(/https?:\/\/[^\s,;，；]+/gi) || [];
  const pmidMatches = String(value || "").match(/\bPMID\s*:?\s*\d{6,9}\b/gi) || [];
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

export function extractLiteratureFreeText(value) {
  const doiMatches = String(value || "").match(/10\.\d{4,9}\/[^\s,;，；]+/gi) || [];
  const urlMatches = String(value || "").match(/https?:\/\/[^\s,;，；]+/gi) || [];
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

export function extractDoiFromText(value) {
  const match = String(value || "").match(/10\.\d{4,9}\/[^\s,;，；]+/i);
  return match ? match[0].replace(/[.)\]}]+$/g, "") : "";
}

export function extractPmidFromText(value) {
  const text = String(value || "");
  const pubmedUrlMatch = text.match(/pubmed\.ncbi\.nlm\.nih\.gov\/(\d{6,9})/i);
  if (pubmedUrlMatch) return pubmedUrlMatch[1];
  const legacyUrlMatch = text.match(/ncbi\.nlm\.nih\.gov\/pubmed\/(\d{6,9})/i);
  if (legacyUrlMatch) return legacyUrlMatch[1];
  const pmidMatch = text.match(/\bPMID\s*:?\s*(\d{6,9})\b/i);
  return pmidMatch ? pmidMatch[1] : "";
}

export function literatureLinkToReference(entry) {
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

export function pdfToReference(file) {
  return {
    title: file.name.replace(/\.(pdf|docx)$/i, ""),
    source: file.name,
    relevance: "用户上传文件，需要进行文献分析。",
    branch_name: "文件上传",
  };
}

export function isResearchDocument(file) {
  return /\.pdf$/i.test(file.name) ||
    /\.docx$/i.test(file.name) ||
    file.type === "application/pdf" ||
    file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
}

export function filterSupportedUploadFiles(files, report) {
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

export function pdfFileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

export function readableTitleFromUrl(url) {
  try {
    const parsed = new URL(url);
    const part = decodeURIComponent(parsed.pathname.split("/").filter(Boolean).pop() || parsed.hostname);
    return part.replace(/[-_]+/g, " ") || parsed.hostname;
  } catch (error) {
    return url;
  }
}

export function referenceStableKey(reference) {
  return String(reference.dedupe_key || reference.doi || reference.pmid || reference.arxiv_id || reference.source || reference.title || "").toLowerCase();
}

export function searchCandidateToAnalysisReference(reference) {
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

export function referenceIdentifierText(reference) {
  if (reference.doi) return `DOI: ${reference.doi}`;
  if (reference.arxiv_id) return `arXiv: ${reference.arxiv_id}`;
  if (reference.pmid) return `PMID: ${reference.pmid}`;
  return reference.source || "无稳定 ID";
}

export function buildLiteratureTopic(entries, pdfFiles, userContext = "", source = "direct", searchQuery = "") {
  const context = String(userContext || "").trim();
  if (context) return context.slice(0, 400);
  if (source === "search" && searchQuery.trim()) return searchQuery.trim().slice(0, 400);
  if (entries.length) return entries.map((entry) => entry.value).slice(0, 3).join(" ");
  if (pdfFiles.length) return pdfFiles.map((file) => file.name).slice(0, 3).join(" ");
  return "literature-analysis";
}

export function buildLiteratureUserContext(userContext, fallback) {
  const context = String(userContext || "").trim();
  if (!context) return fallback;
  return `${fallback}\n\nUser-provided text context or instructions:\n${context}`;
}

export function innovationCellValue(row) {
  return row.what_is_new || row.innovation_point || row.innovation || row.strengths || "";
}

export function publicCellText(value, unclearLabel = "未明确说明") {
  const text = String(value == null ? "" : value).trim();
  if (!text || /^unclear$/i.test(text)) return unclearLabel;
  return text.replace(/\bunclear\b/gi, unclearLabel);
}

export function sourceLabel(source, forceFile = false) {
  return forceFile || !/^https?:\/\//i.test(source || "") ? `文件：${source}` : `来源：${source}`;
}

export function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatSelectedFilesSummary(files) {
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  return `已添加 ${files.length} 个文件，约 ${formatFileSize(totalSize)}`;
}

export function assertUploadSize(files) {
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  if (totalSize > maxUploadBytes) {
    throw new Error(`上传文件总大小约 ${formatFileSize(totalSize)}，超过 ${formatFileSize(maxUploadBytes)} 上限。请减少文件数量，少量多次上传。`);
  }
}

export function safeFileName(value) {
  return (value || "literature-analysis")
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80) || "literature-analysis";
}

export function linkParts(title, source, uploadedFilename = "") {
  const titleText = String(title || "未命名").trim();
  const sourceText = String(source || "").trim();
  const fileText = String(uploadedFilename || "").trim();
  const displaySource = fileText || sourceText;
  const sourceLine = displaySource && displaySource.toLowerCase() !== titleText.toLowerCase()
    ? sourceLabel(displaySource, Boolean(fileText))
    : "";
  return {
    title: titleText,
    href: /^https?:\/\//i.test(sourceText) ? sourceText : "",
    sourceLine,
  };
}
