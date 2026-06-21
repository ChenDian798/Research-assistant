import { apiPath, readJsonResponse } from "./api.js";
import { publicCellText, safeFileName, toStringList } from "./formatters.js";

export function exportAnalysisDocument({ format, rows, summary, topic, onStatus, t }) {
  if (!rows.length && !summary) return;
  const markdown = buildAnalysisMarkdown(rows, summary);
  const reportTitle = analysisReportTitle(topic);
  const baseName = safeFileName(reportTitle);
  if (format === "pdf") {
    downloadPdfDocument({
      title: reportTitle,
      markdown,
      filename: `${baseName}.pdf`,
      onStatus,
      t,
    });
    return;
  }
  const extension = format === "txt" ? "txt" : "md";
  const label = extension.toUpperCase();
  downloadText(`${baseName}.${extension}`, markdown);
  onStatus(translate("export.downloaded", t, { label }), false);
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  downloadBlob(filename, blob);
}

async function downloadPdfDocument({ title, markdown, filename, onStatus, t }) {
  onStatus(translate("export.generatingPdf", t), false);
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
    onStatus(translate("export.pdfDownloaded", t), false);
  } catch (error) {
    onStatus(translate("export.pdfFailed", t, { message: error.message }), true);
  }
}

function translate(key, t, params) {
  if (typeof t === "function") return t(key, params);
  const fallback = {
    "export.downloaded": ({ label }) => `文献分析 ${label} 已下载。`,
    "export.generatingPdf": "正在生成 PDF...",
    "export.pdfDownloaded": "PDF 已下载。",
    "export.pdfFailed": ({ message }) => `PDF 导出失败：${message}`,
  }[key];
  return typeof fallback === "function" ? fallback(params || {}) : fallback;
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

function exportCellValue(value, fallback = "unclear") {
  const items = toStringList(value);
  if (items.length) return items.map((item) => publicCellText(item, fallback)).join("; ");
  const text = String(value == null ? "" : value).trim();
  return publicCellText(text || fallback, fallback);
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
