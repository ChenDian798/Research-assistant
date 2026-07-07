import { useEffect, useMemo, useRef, useState } from "react";
import Topbar from "./components/Topbar.jsx";
import HomeView from "./views/HomeView.jsx";
import DirectAnalysisView from "./views/DirectAnalysisView.jsx";
import StandaloneSearchView from "./views/StandaloneSearchView.jsx";
import NoveltyCheckView from "./views/NoveltyCheckView.jsx";
import SearchFlowView from "./views/SearchFlowView.jsx";
import ResultsView from "./views/ResultsView.jsx";
import HistoryView from "./views/HistoryView.jsx";
import {
  buildLiteratureTopic,
  createEmptyAnalysisResult,
  extractLiteratureFreeText,
  filterSupportedUploadFiles,
  formatFileSize,
  literatureLinkToReference,
  maxLiteraturePdfFiles,
  maxUploadBytes,
  normalizeLiteratureSummary,
  parseLiteratureLinkInput,
  pdfFileKey,
  pdfToReference,
  referenceStableKey,
  searchCandidateToAnalysisReference,
} from "./lib/formatters.js";
import {
  deleteHistoryEntry,
  fetchHistoryEntries,
  fetchHistoryEntry,
  readJsonResponse,
  submitCombinedLiteratureAnalysis,
  submitLinkLiteratureAnalysis,
  submitLiteratureSearchRequest,
  submitNoveltyCheckRequest,
  waitForJob,
} from "./lib/api.js";
import { exportAnalysisDocument } from "./lib/export.js";
import { createTranslator, getInitialLanguage, saveLanguage } from "./lib/i18n.js";
import {
  allowedViews,
  defaultNoveltyForm,
  defaultSearchForm,
  hasStoredAnalysisResult,
  historySearchCandidates,
  isActiveHistoryEntry,
  isHistorySummaryOnly,
  mergeHistorySummaryIntoDetail,
  storedAnalysisResult,
  viewFromHash,
} from "./lib/appState.js";

export default function App() {
  const [language, setLanguage] = useState(getInitialLanguage);
  const [currentView, setCurrentView] = useState(viewFromHash);
  const [searchStep, setSearchStep] = useState(1);
  const [doiInput, setDoiInput] = useState("");
  const [selectedPdfFiles, setSelectedPdfFiles] = useState([]);
  const [directStatus, setDirectStatus] = useState("");
  const [directStatusError, setDirectStatusError] = useState(false);
  const [searchStatus, setSearchStatus] = useState("");
  const [searchStatusError, setSearchStatusError] = useState(false);
  const [analysisRunning, setAnalysisRunning] = useState({ direct: false, search: false });
  const [activeAnalysisSource, setActiveAnalysisSource] = useState("direct");
  const [analysisMode, setAnalysisMode] = useState("idle");
  const [analysisError, setAnalysisError] = useState("");
  const [analysisResultsBySource, setAnalysisResultsBySource] = useState({
    direct: createEmptyAnalysisResult(),
    search: createEmptyAnalysisResult(),
  });
  const analysisRunIds = useRef({ direct: 0, search: 0 });
  const searchRunId = useRef(0);
  const standaloneSearchRunId = useRef(0);
  const noveltyRunId = useRef(0);

  const [searchForm, setSearchForm] = useState(defaultSearchForm);
  const [searchLoading, setSearchLoading] = useState(false);
  const [candidatePayload, setCandidatePayload] = useState({ rejected_count: 0, errors: {} });
  const [searchCandidateReferences, setSearchCandidateReferences] = useState([]);
  const [selectedCandidateIds, setSelectedCandidateIds] = useState(new Set());
  const [stagedAnalysisReferences, setStagedAnalysisReferences] = useState([]);
  const [activeSearchHistoryId, setActiveSearchHistoryId] = useState("");
  const [searchAnalysisQueued, setSearchAnalysisQueued] = useState(false);
  const [hasSearchResult, setHasSearchResult] = useState(false);
  const [standaloneSearchForm, setStandaloneSearchForm] = useState(defaultSearchForm);
  const [standaloneSearchStatus, setStandaloneSearchStatus] = useState("");
  const [standaloneSearchStatusError, setStandaloneSearchStatusError] = useState(false);
  const [standaloneSearchLoading, setStandaloneSearchLoading] = useState(false);
  const [standaloneCandidatePayload, setStandaloneCandidatePayload] = useState({ rejected_count: 0, errors: {} });
  const [standaloneSearchCandidateReferences, setStandaloneSearchCandidateReferences] = useState([]);
  const [standaloneSelectedCandidateIds, setStandaloneSelectedCandidateIds] = useState(new Set());
  const [standaloneActiveSearchHistoryId, setStandaloneActiveSearchHistoryId] = useState("");
  const [standaloneHasSearchResult, setStandaloneHasSearchResult] = useState(false);
  const [noveltyForm, setNoveltyForm] = useState(defaultNoveltyForm);
  const [noveltyStatus, setNoveltyStatus] = useState("");
  const [noveltyStatusError, setNoveltyStatusError] = useState(false);
  const [noveltyRunning, setNoveltyRunning] = useState(false);
  const [noveltyResult, setNoveltyResult] = useState(null);
  const [noveltyActiveHistoryId, setNoveltyActiveHistoryId] = useState("");
  const [exportFormat, setExportFormat] = useState("md");
  const [historyEntries, setHistoryEntries] = useState([]);
  const [selectedHistoryEntry, setSelectedHistoryEntry] = useState(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const selectedHistoryIdRef = useRef("");
  const selectedHistoryEntryRef = useRef(null);
  const historyListRunId = useRef(0);
  const historyDetailRunId = useRef(0);
  const t = useMemo(() => createTranslator(language), [language]);

  useEffect(() => {
    document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  }, [language]);

  useEffect(() => {
    const onHashChange = () => setCurrentView(viewFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = (view, updateHash = true) => {
    const nextView = allowedViews.includes(view) ? view : "home";
    setCurrentView(nextView);
    if (updateHash && window.location.hash !== `#${nextView}`) {
      history.pushState(null, "", `#${nextView}`);
    }
  };

  const showSearchStep = (step) => {
    navigate("search");
    setSearchStep(Number(step) || 1);
  };

  useEffect(() => {
    selectedHistoryIdRef.current = selectedHistoryId;
  }, [selectedHistoryId]);

  useEffect(() => {
    selectedHistoryEntryRef.current = selectedHistoryEntry;
  }, [selectedHistoryEntry]);

  const loadHistoryEntryDetail = async (historyId, options = {}) => {
    if (!historyId) return;
    const runId = ++historyDetailRunId.current;
    if (options.summary) {
      setSelectedHistoryEntry((entry) => (entry?.id === historyId && !isHistorySummaryOnly(entry) ? entry : options.summary));
    }
    if (options.loading !== false) setHistoryLoading(true);
    setHistoryError("");
    try {
      const entry = await fetchHistoryEntry(historyId);
      if (runId !== historyDetailRunId.current) return;
      selectedHistoryEntryRef.current = entry;
      setSelectedHistoryEntry(entry);
      setSelectedHistoryId(entry.id);
      selectedHistoryIdRef.current = entry.id;
    } catch (error) {
      if (runId === historyDetailRunId.current) setHistoryError(error.message);
    } finally {
      if (runId === historyDetailRunId.current && options.loading !== false) setHistoryLoading(false);
    }
  };

  const loadHistory = async (selectId = "", options = {}) => {
    const runId = ++historyListRunId.current;
    if (options.loading !== false) setHistoryLoading(true);
    setHistoryError("");
    try {
      const entries = await fetchHistoryEntries();
      if (runId !== historyListRunId.current) return;
      const targetId = selectId || selectedHistoryIdRef.current || entries[0]?.id || "";
      const targetSummary = entries.find((entry) => entry.id === targetId) || null;
      setHistoryEntries(entries);
      if (targetId) {
        const currentDetail = selectedHistoryEntryRef.current;
        const shouldFetchDetail = Boolean(
          options.forceDetail ||
          selectId ||
          !currentDetail ||
          currentDetail.id !== targetId ||
          isHistorySummaryOnly(currentDetail) ||
          isActiveHistoryEntry(currentDetail) ||
          isActiveHistoryEntry(targetSummary),
        );
        setSelectedHistoryId(targetId);
        selectedHistoryIdRef.current = targetId;
        if (shouldFetchDetail) {
          loadHistoryEntryDetail(targetId, { summary: targetSummary, loading: options.loading !== false });
        } else if (targetSummary) {
          setSelectedHistoryEntry((entry) => mergeHistorySummaryIntoDetail(entry, targetSummary));
        }
      } else {
        setSelectedHistoryEntry(null);
        setSelectedHistoryId("");
      }
    } catch (error) {
      if (runId === historyListRunId.current) setHistoryError(error.message);
    } finally {
      if (runId === historyListRunId.current && options.loading !== false) setHistoryLoading(false);
    }
  };

  useEffect(() => {
    if (currentView !== "history") return undefined;
    loadHistory("", { forceDetail: true });
    const timer = window.setInterval(() => loadHistory(), 5000);
    return () => window.clearInterval(timer);
  }, [currentView]);

  const handleSelectHistoryEntry = async (historyId) => {
    if (!historyId) return;
    selectedHistoryIdRef.current = historyId;
    const summary = historyEntries.find((entry) => entry.id === historyId) || null;
    setSelectedHistoryId(historyId);
    loadHistoryEntryDetail(historyId, { summary });
  };

  const handleDeleteHistoryEntry = async (entry) => {
    if (!entry?.id) return;
    const confirmed = window.confirm(t("history.deleteConfirm", { title: entry.title || t("history.untitled") }));
    if (!confirmed) return;
    const deleteIds = Array.from(new Set([entry.id, entry.analysis?.id].filter(Boolean)));
    setHistoryLoading(true);
    setHistoryError("");
    try {
      await Promise.all(deleteIds.map((historyId) => deleteHistoryEntry(historyId)));
      const entries = await fetchHistoryEntries();
      setHistoryEntries(entries);
      const nextEntry = entries.find((item) => !deleteIds.includes(item.id)) || null;
      if (nextEntry) {
        const fullEntry = await fetchHistoryEntry(nextEntry.id);
        setSelectedHistoryEntry(fullEntry);
        setSelectedHistoryId(fullEntry.id);
      } else {
        setSelectedHistoryEntry(null);
        setSelectedHistoryId("");
      }
    } catch (error) {
      setHistoryError(error.message);
    } finally {
      setHistoryLoading(false);
    }
  };

  const resolveSearchHistoryEntry = async (entry) => {
    if (!(entry?.kind === "literature_search" || entry?.kind === "search_flow") || !entry?.id) {
      return entry;
    }
    if (historySearchCandidates(entry).length) return entry;
    try {
      const detailedEntry = await fetchHistoryEntry(entry.id);
      return {
        ...entry,
        ...detailedEntry,
        analysis: detailedEntry.analysis || entry.analysis,
      };
    } catch (error) {
      setHistoryError(error.message);
      return entry;
    }
  };

  const handleResubmitHistoryEntry = async (entry) => {
    if (entry?.kind === "literature_search" || entry?.kind === "search_flow") {
      entry = await resolveSearchHistoryEntry(entry);
    }
    const request = entry?.request || {};
    if (entry?.kind === "novelty_check") {
      setNoveltyForm((current) => ({
        ...current,
        innovationText: request.innovation_text || entry.title || "",
        searchMode: request.search_mode || current.searchMode || "auto",
        year: request.year || "",
        sources: String(request.sources || "")
          .split(",")
          .map((source) => source.trim())
          .filter(Boolean),
        includeFilteredReferences: Boolean(request.include_filtered_references),
      }));
      setNoveltyRunning(false);
      setNoveltyResult(entry.result && Object.keys(entry.result).length ? entry.result : null);
      setNoveltyActiveHistoryId(entry.id || "");
      setNoveltyStatus(t("history.resubmitNoveltyLoaded"));
      setNoveltyStatusError(false);
      navigate("novelty");
      return;
    }

    if (entry?.kind === "literature_search" || entry?.kind === "search_flow") {
      const result = entry?.result || {};
      const references = historySearchCandidates(entry);
      setStandaloneSearchForm((current) => ({
        ...current,
        query: request.query || result.query || entry.title || "",
        searchMode: request.search_mode || result.search_mode || current.searchMode || "auto",
        year: request.year || "",
        limit: String(request.max_results_per_source || current.limit || "5"),
        sources: String(request.sources || "")
          .split(",")
          .map((source) => source.trim())
          .filter(Boolean),
        includeNeedsReview: request.include_needs_review !== false,
      }));
      setStandaloneSearchLoading(false);
      setStandaloneSearchCandidateReferences(references);
      setStandaloneSelectedCandidateIds(new Set(references
        .filter((reference) => reference.candidate_group === "qualified")
        .map((reference) => reference.candidate_id)));
      setStandaloneCandidatePayload(references.length ? result : { rejected_count: 0, errors: {} });
      setStandaloneHasSearchResult(Boolean(references.length || entry?.status === "done" || entry?.stage === "Search complete"));
      setStandaloneActiveSearchHistoryId(entry.id || "");
      setStandaloneSearchStatus(references.length || entry?.status === "done" || entry?.stage === "Search complete"
        ? t("history.continueSearchLoaded", { count: references.length })
        : t("history.resubmitSearchLoaded"));
      setStandaloneSearchStatusError(false);
      navigate("standaloneSearch");
      return;
    }

    if (entry?.kind === "search_analysis") {
      const references = Array.isArray(request.references) ? request.references : [];
      setSearchForm((current) => ({
        ...current,
        query: request.topic || entry.title || current.query,
      }));
      setSearchCandidateReferences([]);
      setSelectedCandidateIds(new Set());
      setStagedAnalysisReferences(references);
      setCandidatePayload({ rejected_count: 0, errors: {} });
      setHasSearchResult(false);
      setSearchStatus(t("history.resubmitAnalysisLoaded", { count: references.length }));
      setSearchStatusError(false);
      setSearchAnalysisQueued(false);
      setSearchStep(3);
      navigate("search");
      return;
    }

    if (entry?.kind === "direct_analysis") {
      const references = Array.isArray(request.references) ? request.references : [];
      const linkText = references
        .map((reference) => reference.source || reference.doi || reference.pmid || reference.arxiv_id || "")
        .filter((value) => /^https?:\/\//i.test(value) || /^10\.\d{4,9}\//i.test(value))
        .join("\n");
      const fileCount = Number(request.file_count || 0);
      setDoiInput(fileCount ? linkText : (linkText || request.topic || entry.title || ""));
      setSelectedPdfFiles([]);
      setDirectStatus(fileCount
        ? t("history.resubmitDirectFilesLoaded", {
          fileCount,
          referenceCount: Number(request.reference_count || references.length || 0),
        })
        : t("history.resubmitDirectLoaded"));
      setDirectStatusError(false);
      navigate("direct");
    }
  };

  const handleContinueHistoryEntry = async (entry) => {
    entry = await resolveSearchHistoryEntry(entry);
    const request = entry?.request || {};
    const result = entry?.result || {};
    if (entry?.kind === "literature_search" || entry?.kind === "search_flow") {
      const references = historySearchCandidates(entry);
      const flowAnalysis = entry?.analysis && typeof entry.analysis === "object" ? entry.analysis : null;
      const hasFlowAnalysisResult = hasStoredAnalysisResult(flowAnalysis);
      analysisRunIds.current.search += 1;
      setActiveSearchHistoryId(entry.id || "");
      setActiveAnalysisSource("search");
      setAnalysisMode(hasFlowAnalysisResult ? "done" : "idle");
      setAnalysisError("");
      setSearchForm((current) => ({
        ...current,
        query: request.query || result.query || entry.title || "",
        searchMode: request.search_mode || result.search_mode || current.searchMode || "auto",
        year: request.year || "",
        limit: String(request.max_results_per_source || current.limit || "5"),
        sources: String(request.sources || "")
          .split(",")
          .map((source) => source.trim())
          .filter(Boolean),
        includeNeedsReview: request.include_needs_review !== false,
      }));
      setSearchCandidateReferences(references);
      setSelectedCandidateIds(new Set(references
        .filter((reference) => reference.candidate_group === "qualified")
        .map((reference) => reference.candidate_id)));
      setStagedAnalysisReferences([]);
      setCandidatePayload(result);
      setHasSearchResult(true);
      setSearchStatus(t("history.continueSearchLoaded", { count: references.length }));
      setSearchStatusError(false);
      setSearchAnalysisQueued(false);
      setAnalysisRunning((current) => ({ ...current, search: false }));
      if (hasFlowAnalysisResult) {
        const restoredResult = storedAnalysisResult(flowAnalysis, request.query || result.query || entry.title || "literature-search-analysis");
        updateResult("search", restoredResult);
        setExportFormat(restoredResult.rows.length ? "md" : "txt");
        setSearchStatus(restoredResult.reviewNeededDocuments.length
          ? t("status.analysisCompleteWithReview", {
            count: restoredResult.rows.length,
            reviewCount: restoredResult.reviewNeededDocuments.length,
          })
          : t("status.analysisComplete", { count: restoredResult.rows.length }));
        setSearchStep(4);
        navigate("results");
        return;
      }
      updateResult("search", createEmptyAnalysisResult());
      setSearchStep(references.length || entry?.status === "done" || entry?.stage === "Search complete" ? 2 : 1);
      navigate("search");
      return;
    }
    handleResubmitHistoryEntry(entry);
  };

  const handleLanguageChange = (nextLanguage) => {
    setLanguage(saveLanguage(nextLanguage));
  };

  const linkSummary = useMemo(() => {
    const entries = parseLiteratureLinkInput(doiInput);
    const userContext = extractLiteratureFreeText(doiInput);
    if (!entries.length && !userContext) return t("summary.noLinks");
    const counts = entries.reduce((acc, entry) => {
      acc[entry.type] = (acc[entry.type] || 0) + 1;
      return acc;
    }, {});
    const parts = [
      counts.doi ? t("summary.doiCount", { count: counts.doi }) : "",
      counts.pmid ? t("summary.pmidCount", { count: counts.pmid }) : "",
      counts.url ? t("summary.urlCount", { count: counts.url }) : "",
      userContext ? t("summary.hasText") : "",
    ].filter(Boolean);
    return t("summary.detected", { parts: parts.join(language === "en" ? ", " : "，") });
  }, [doiInput, language, t]);

  const pdfSummary = useMemo(() => {
    if (!selectedPdfFiles.length) return t("summary.noFiles");
    const totalSize = selectedPdfFiles.reduce((sum, file) => sum + file.size, 0);
    const suffix = totalSize > maxUploadBytes ? t("summary.overLimit", { size: formatFileSize(maxUploadBytes) }) : "";
    return `${t("summary.filesAdded", { count: selectedPdfFiles.length, size: formatFileSize(totalSize) })}${suffix}`;
  }, [selectedPdfFiles, t]);

  const activeResult = analysisResultsBySource[activeAnalysisSource] || createEmptyAnalysisResult();
  const searchAnalysisResult = analysisResultsBySource.search || createEmptyAnalysisResult();
  const directAnalysisResult = analysisResultsBySource.direct || createEmptyAnalysisResult();
  const hasDirectAnalysisResult = Boolean(
    directAnalysisResult.rows.length ||
    directAnalysisResult.summary ||
    directAnalysisResult.reviewNeededDocuments.length
  );
  const hasSearchAnalysisResult = Boolean(
    searchAnalysisResult.rows.length ||
    searchAnalysisResult.summary ||
    searchAnalysisResult.reviewNeededDocuments.length
  );
  const candidateMeta = useMemo(() => {
    return buildSearchCandidateMeta(candidatePayload, searchCandidateReferences.length);
  }, [candidatePayload, searchCandidateReferences.length, t]);
  const standaloneCandidateMeta = useMemo(() => {
    return buildSearchCandidateMeta(standaloneCandidatePayload, standaloneSearchCandidateReferences.length);
  }, [standaloneCandidatePayload, standaloneSearchCandidateReferences.length, t]);

  function buildSearchCandidateMeta(payload, count) {
    const rejectedCount = Number(payload.rejected_count || 0);
    const errors = payload.errors || {};
    const errorText = Object.entries(errors).map(([source, message]) => `${source}: ${message}`).join("；");
    const mode = payload.search_mode || payload.requested_search_mode || "";
    return [
      t("candidate.meta", { count }),
      mode ? t("candidate.searchMode", { mode: t(`search.mode.${mode}`) }) : "",
      rejectedCount ? t("candidate.rejected", { count: rejectedCount }) : "",
      errorText ? t("candidate.sourceHint", { text: errorText }) : "",
    ].filter(Boolean).join(" · ");
  }

  function updateResult(source, result) {
    setAnalysisResultsBySource((current) => ({ ...current, [source]: result }));
  }

  function handleFilesSelected(rawFiles) {
    const files = filterSupportedUploadFiles(rawFiles, (message) => {
      setDirectStatus(message);
      setDirectStatusError(false);
    });
    if (selectedPdfFiles.length + files.length > maxLiteraturePdfFiles) {
      setSelectedPdfFiles([]);
      setDirectStatus(t("status.tooManyFiles", { max: maxLiteraturePdfFiles }));
      setDirectStatusError(false);
      return;
    }
    if (!files.length) return;
    setSelectedPdfFiles((current) => {
      const existing = new Set(current.map(pdfFileKey));
      const additions = files.filter((file) => {
        const key = pdfFileKey(file);
        if (existing.has(key)) return false;
        existing.add(key);
        return true;
      });
      return [...current, ...additions];
    });
  }

  function handleRemoveSelectedFile(fileToRemove) {
    const keyToRemove = pdfFileKey(fileToRemove);
    setSelectedPdfFiles((current) => current.filter((file) => pdfFileKey(file) !== keyToRemove));
  }

  function buildAnalysisRequest(source) {
    if (source === "search") {
      const references = stagedAnalysisReferences.map((reference) => ({ ...reference }));
      if (!references.length) {
        return { error: t("analysis.needSearchSelection"), step: 3 };
      }
      const topic = (searchForm.query.trim() || "literature-search-analysis").slice(0, 400);
      return { source, references, previewReferences: references, pdfFiles: [], userContext: "", topic };
    }

    const entries = parseLiteratureLinkInput(doiInput);
    const userContext = extractLiteratureFreeText(doiInput);
    const pdfFiles = selectedPdfFiles;
    if (!entries.length && !pdfFiles.length && !userContext) {
      return { error: t("analysis.needDirectInput") };
    }
    const linkReferences = entries.map(literatureLinkToReference);
    return {
      source,
      references: linkReferences,
      previewReferences: [...linkReferences, ...pdfFiles.map(pdfToReference)],
      pdfFiles,
      userContext,
      topic: buildLiteratureTopic(entries, pdfFiles, userContext, source, searchForm.query),
    };
  }

  async function submitLiteratureAnalysisFromSource(source) {
    const setStatus = source === "search" ? setSearchStatus : setDirectStatus;
    const setStatusError = source === "search" ? setSearchStatusError : setDirectStatusError;
    const request = buildAnalysisRequest(source);
    if (request.error) {
      setStatus(request.error);
      setStatusError(false);
      if (source === "search") showSearchStep(request.step || 3);
      return;
    }

    if (source === "search") {
      const runId = ++analysisRunIds.current.search;
      setStatus(t("status.queueingSearchAnalysis"));
      setStatusError(false);
      setSearchAnalysisQueued(false);
      setActiveAnalysisSource("search");
      setAnalysisMode("loading");
      setAnalysisError("");
      updateResult("search", { ...createEmptyAnalysisResult(request.topic), displayReferences: request.previewReferences });
      setAnalysisRunning((current) => ({ ...current, search: true }));
      try {
        const response = await submitLinkLiteratureAnalysis(request.references, request.userContext, request.topic, source, activeSearchHistoryId, language);
        let payload = await readJsonResponse(response);
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
        loadHistory(payload.history_id);
        setActiveSearchHistoryId(payload.history_id || activeSearchHistoryId);
        setSearchAnalysisQueued(true);
        setSearchStatus(t("status.searchAnalysisQueued"));
        setSearchStatusError(false);
        setSearchStep(4);
        navigate("search");
        if (payload.job_id) {
          payload = await waitForJob("/api/literature-analysis", payload.job_id, (message) => {
            if (analysisRunIds.current.search === runId) setSearchStatus(message);
          }, t);
        }
        if (analysisRunIds.current.search !== runId) return;

        const rows = Array.isArray(payload.rows) ? payload.rows : [];
        const summary = normalizeLiteratureSummary(payload.summary);
        const references = Array.isArray(payload.references) && payload.references.length
          ? payload.references
          : request.previewReferences;
        const reviewNeededDocuments = Array.isArray(payload.review_needed_documents) ? payload.review_needed_documents : [];
        updateResult("search", {
          rows,
          summary,
          topic: request.topic,
          displayReferences: references,
          reviewNeededDocuments,
        });
        setAnalysisMode("done");
        setExportFormat(rows.length ? "md" : "txt");
        setSearchAnalysisQueued(false);
        setSearchStatus(reviewNeededDocuments.length
          ? t("status.analysisCompleteWithReview", { count: rows.length, reviewCount: reviewNeededDocuments.length })
          : t("status.analysisComplete", { count: rows.length }));
        setSearchStatusError(false);
        loadHistory(payload.history_id || activeSearchHistoryId);
      } catch (error) {
        if (analysisRunIds.current.search !== runId) return;
        setSearchAnalysisQueued(false);
        setAnalysisMode("error");
        setAnalysisError(error.message);
        updateResult("search", { ...createEmptyAnalysisResult(request.topic), displayReferences: request.previewReferences });
        setSearchStatus(`${t("status.searchAnalysisFailed")} - ${error.message}`);
        setSearchStatusError(true);
      } finally {
        if (analysisRunIds.current.search === runId) {
          setAnalysisRunning((current) => ({ ...current, search: false }));
        }
      }
      return;
    }

    if (analysisRunning[source]) {
      setActiveAnalysisSource(source);
      setAnalysisMode("loading");
      if (source === "search") setSearchStep(4);
      navigate("results");
      return;
    }

    const runId = ++analysisRunIds.current[source];
    setActiveAnalysisSource(source);
    setAnalysisMode("loading");
    setAnalysisError("");
    navigate("results");
    updateResult(source, { ...createEmptyAnalysisResult(request.topic), displayReferences: request.previewReferences });
    setAnalysisRunning((current) => ({ ...current, [source]: true }));
    setStatus(source === "search" ? t("status.startingSearchAnalysis") : t("status.startingDirectAnalysis"));
    setStatusError(false);

    try {
      const response = request.pdfFiles.length
        ? await submitCombinedLiteratureAnalysis(request.references, request.pdfFiles, request.userContext, request.topic, source, language)
        : await submitLinkLiteratureAnalysis(request.references, request.userContext, request.topic, source, "", language);
      let payload = await readJsonResponse(response);
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      const acceptedReferences = Array.isArray(payload.references) ? payload.references : [];
      const reviewNeededDocuments = Array.isArray(payload.review_needed_documents) ? payload.review_needed_documents : [];
      if (payload.job_id) {
        loadHistory(payload.history_id);
        payload = await waitForJob("/api/literature-analysis", payload.job_id, (message) => {
          if (analysisRunIds.current[source] === runId) setStatus(message);
        }, t);
      }
      if (analysisRunIds.current[source] !== runId) return;

      const rows = Array.isArray(payload.rows) ? payload.rows : [];
      const summary = normalizeLiteratureSummary(payload.summary);
      const result = {
        rows,
        summary,
        topic: request.topic,
        displayReferences: acceptedReferences.length ? acceptedReferences : request.previewReferences,
        reviewNeededDocuments,
      };
      updateResult(source, result);
      setAnalysisMode("done");
      setExportFormat(rows.length ? "md" : "txt");
      setStatus(reviewNeededDocuments.length
        ? t("status.analysisCompleteWithReview", { count: rows.length, reviewCount: reviewNeededDocuments.length })
        : t("status.analysisComplete", { count: rows.length }));
      if (source === "search") {
        setSearchStep(4);
        navigate("results");
      }
      if (source === "direct") setSelectedPdfFiles([]);
    } catch (error) {
      if (analysisRunIds.current[source] !== runId) return;
      updateResult(source, { ...createEmptyAnalysisResult(request.topic), displayReferences: request.previewReferences });
      setAnalysisMode("error");
      setAnalysisError(error.message);
      setStatus(`${source === "search" ? t("status.searchAnalysisFailed") : t("status.directAnalysisFailed")} - ${error.message}`);
      setStatusError(true);
    } finally {
      if (analysisRunIds.current[source] === runId) {
        setAnalysisRunning((current) => ({ ...current, [source]: false }));
      }
    }
  }

  function handleSearchFormChange(field, value) {
    setSearchForm((current) => ({ ...current, [field]: value }));
  }

  function handleToggleSource(value, checked) {
    setSearchForm((current) => ({
      ...current,
      sources: checked
        ? [...current.sources, value]
        : current.sources.filter((source) => source !== value),
    }));
  }

  function handleStandaloneSearchFormChange(field, value) {
    setStandaloneSearchForm((current) => ({ ...current, [field]: value }));
  }

  function handleStandaloneToggleSource(value, checked) {
    setStandaloneSearchForm((current) => ({
      ...current,
      sources: checked
        ? [...current.sources, value]
        : current.sources.filter((source) => source !== value),
    }));
  }

  function handleNoveltyFormChange(field, value) {
    setNoveltyForm((current) => ({ ...current, [field]: value }));
  }

  function handleNoveltyToggleSource(value, checked) {
    setNoveltyForm((current) => ({
      ...current,
      sources: checked
        ? [...current.sources, value]
        : current.sources.filter((source) => source !== value),
    }));
  }

  function applySearchResultPayload(payload) {
    const { references, selectedIds, qualified, needsReview } = normalizeSearchResultPayload(payload);
    setSearchCandidateReferences(references);
    setSelectedCandidateIds(selectedIds);
    setCandidatePayload(payload);
    setHasSearchResult(true);
    setSearchStatus(t("status.searchComplete", {
      qualified: qualified.length,
      needsReview: needsReview.length,
      rejected: payload.rejected_count || 0,
    }));
    setSearchStatusError(false);
    loadHistory(payload.history_id);
    setActiveSearchHistoryId(payload.history_id || "");
  }

  function applyStandaloneSearchResultPayload(payload) {
    const { references, selectedIds, qualified, needsReview } = normalizeSearchResultPayload(payload);
    setStandaloneSearchCandidateReferences(references);
    setStandaloneSelectedCandidateIds(selectedIds);
    setStandaloneCandidatePayload(payload);
    setStandaloneHasSearchResult(true);
    setStandaloneSearchStatus(t("status.searchComplete", {
      qualified: qualified.length,
      needsReview: needsReview.length,
      rejected: payload.rejected_count || 0,
    }));
    setStandaloneSearchStatusError(false);
    loadHistory(payload.history_id);
    setStandaloneActiveSearchHistoryId(payload.history_id || "");
  }

  function normalizeSearchResultPayload(payload) {
    const qualified = (payload.qualified_references || []).map((reference) => ({
      ...reference,
      candidate_group: "qualified",
    }));
    const needsReview = (payload.needs_review_references || []).map((reference) => ({
      ...reference,
      candidate_group: "needs_review",
    }));
    const rejected = (payload.rejected_references || []).map((reference) => ({
      ...reference,
      candidate_group: "rejected",
    }));
    const references = [...qualified, ...needsReview, ...rejected].map((reference, index) => ({
      ...reference,
      candidate_id: reference.dedupe_key || reference.source || reference.doi || reference.title || `candidate-${index}`,
    }));
    const selectedIds = new Set(qualified.map((reference, index) => (
      reference.dedupe_key || reference.source || reference.doi || reference.title || `candidate-${index}`
    )));
    return { references, selectedIds, qualified, needsReview, rejected };
  }

  function showSearchTaskTarget(targetView, step = 2) {
    if (targetView === "standaloneSearch") {
      navigate("standaloneSearch");
      return;
    }
    showSearchStep(step);
  }

  async function submitLiteratureSearch(targetView = "search") {
    const query = searchForm.query.trim();
    if (!query) {
      setSearchStatus(t("status.enterSearchQuery"));
      setSearchStatusError(false);
      showSearchTaskTarget(targetView, 1);
      return;
    }
    const sources = searchForm.sources.join(",");
    if (!sources) {
      setSearchStatus(t("status.chooseSource"));
      setSearchStatusError(false);
      showSearchTaskTarget(targetView, 1);
      return;
    }
    const runId = ++searchRunId.current;
    setSearchLoading(true);
    setSearchAnalysisQueued(false);
    setHasSearchResult(true);
    setSearchStatus(t("status.searchingCandidates"));
    setSearchStatusError(false);
    showSearchTaskTarget(targetView, 2);
    try {
      let payload = await submitLiteratureSearchRequest({
        query,
        sources,
        search_mode: searchForm.searchMode || "auto",
        max_results_per_source: Number(searchForm.limit || 5),
        year: searchForm.year.trim(),
        include_needs_review: searchForm.includeNeedsReview,
        append_annotation_record: searchForm.appendAnnotationRecord,
        run_async: true,
      });
      if (payload.status === "queued" && payload.job_id) {
        setActiveSearchHistoryId(payload.history_id || "");
        loadHistory(payload.history_id);
        setSearchCandidateReferences([]);
        setSelectedCandidateIds(new Set());
        setCandidatePayload({ rejected_count: 0, errors: {} });
        setHasSearchResult(true);
        setSearchStatus(t(targetView === "standaloneSearch" ? "standaloneSearch.searchQueued" : "status.searchQueued"));
        setSearchStatusError(false);
        payload = await waitForJob("/api/literature-search", payload.job_id, (message) => {
          if (searchRunId.current === runId) setSearchStatus(message || t("status.searchingCandidates"));
        }, t);
      }
      if (searchRunId.current !== runId) return;
      applySearchResultPayload(payload);
    } catch (error) {
      if (searchRunId.current !== runId) return;
      setSearchCandidateReferences([]);
      setSelectedCandidateIds(new Set());
      setCandidatePayload({ rejected_count: 0, errors: { search: error.message } });
      setSearchStatus(t("status.searchFailed", { message: error.message }));
      setSearchStatusError(true);
      if (error.payload?.history_id) loadHistory(error.payload.history_id);
    } finally {
      if (searchRunId.current === runId) {
        setSearchLoading(false);
        setSearchStep(2);
      }
    }
  }

  async function submitStandaloneLiteratureSearch() {
    const query = standaloneSearchForm.query.trim();
    if (!query) {
      setStandaloneSearchStatus(t("status.enterSearchQuery"));
      setStandaloneSearchStatusError(false);
      navigate("standaloneSearch");
      return;
    }
    const sources = standaloneSearchForm.sources.join(",");
    if (!sources) {
      setStandaloneSearchStatus(t("status.chooseSource"));
      setStandaloneSearchStatusError(false);
      navigate("standaloneSearch");
      return;
    }
    const runId = ++standaloneSearchRunId.current;
    setStandaloneSearchLoading(true);
    setStandaloneHasSearchResult(true);
    setStandaloneSearchStatus(t("status.searchingCandidates"));
    setStandaloneSearchStatusError(false);
    navigate("standaloneSearch");
    try {
      let payload = await submitLiteratureSearchRequest({
        query,
        sources,
        search_mode: standaloneSearchForm.searchMode || "auto",
        max_results_per_source: Number(standaloneSearchForm.limit || 5),
        year: standaloneSearchForm.year.trim(),
        include_needs_review: standaloneSearchForm.includeNeedsReview,
        append_annotation_record: standaloneSearchForm.appendAnnotationRecord,
        run_async: true,
      });
      if (payload.status === "queued" && payload.job_id) {
        setStandaloneActiveSearchHistoryId(payload.history_id || "");
        loadHistory(payload.history_id);
        setStandaloneSearchCandidateReferences([]);
        setStandaloneSelectedCandidateIds(new Set());
        setStandaloneCandidatePayload({ rejected_count: 0, errors: {} });
        setStandaloneHasSearchResult(true);
        setStandaloneSearchStatus(t("standaloneSearch.searchQueued"));
        setStandaloneSearchStatusError(false);
        payload = await waitForJob("/api/literature-search", payload.job_id, (message) => {
          if (standaloneSearchRunId.current === runId) setStandaloneSearchStatus(message || t("status.searchingCandidates"));
        }, t);
      }
      if (standaloneSearchRunId.current !== runId) return;
      applyStandaloneSearchResultPayload(payload);
    } catch (error) {
      if (standaloneSearchRunId.current !== runId) return;
      setStandaloneSearchCandidateReferences([]);
      setStandaloneSelectedCandidateIds(new Set());
      setStandaloneCandidatePayload({ rejected_count: 0, errors: { search: error.message } });
      setStandaloneSearchStatus(t("status.searchFailed", { message: error.message }));
      setStandaloneSearchStatusError(true);
      if (error.payload?.history_id) loadHistory(error.payload.history_id);
    } finally {
      if (standaloneSearchRunId.current === runId) {
        setStandaloneSearchLoading(false);
      }
    }
  }

  async function submitNoveltyCheck() {
    const innovationText = noveltyForm.innovationText.trim();
    if (!innovationText) {
      setNoveltyStatus(t("novelty.needInput"));
      setNoveltyStatusError(false);
      navigate("novelty");
      return;
    }
    const sources = noveltyForm.sources.join(",");
    if (!sources) {
      setNoveltyStatus(t("status.chooseSource"));
      setNoveltyStatusError(false);
      navigate("novelty");
      return;
    }
    const runId = ++noveltyRunId.current;
    setNoveltyRunning(true);
    setNoveltyResult(null);
    setNoveltyStatus(t("novelty.submitting"));
    setNoveltyStatusError(false);
    navigate("novelty");
    try {
      let payload = await submitNoveltyCheckRequest({
        innovation_text: innovationText,
        sources,
        search_mode: noveltyForm.searchMode || "auto",
        year: noveltyForm.year.trim(),
        include_filtered_references: noveltyForm.includeFilteredReferences,
      });
      if (payload.status === "queued" && payload.job_id) {
        setNoveltyActiveHistoryId(payload.history_id || "");
        loadHistory(payload.history_id);
        setNoveltyStatus(t("novelty.queued"));
        payload = await waitForJob("/api/novelty-check", payload.job_id, (message) => {
          if (noveltyRunId.current === runId) setNoveltyStatus(message || t("novelty.running"));
        }, t);
      }
      if (noveltyRunId.current !== runId) return;
      setNoveltyResult(payload);
      setNoveltyActiveHistoryId(payload.history_id || "");
      setNoveltyStatus(t("novelty.complete"));
      setNoveltyStatusError(false);
      loadHistory(payload.history_id);
    } catch (error) {
      if (noveltyRunId.current !== runId) return;
      setNoveltyResult(null);
      setNoveltyStatus(t("novelty.failed", { message: error.message }));
      setNoveltyStatusError(true);
      if (error.payload?.history_id) loadHistory(error.payload.history_id);
    } finally {
      if (noveltyRunId.current === runId) setNoveltyRunning(false);
    }
  }

  function toggleCandidate(candidateId, isSelected) {
    setSelectedCandidateIds((current) => {
      const next = new Set(current);
      if (isSelected) next.add(candidateId);
      else next.delete(candidateId);
      return next;
    });
  }

  function addSelectedSearchReferencesToAnalysis() {
    const existing = new Set(stagedAnalysisReferences.map(referenceStableKey));
    const selected = searchCandidateReferences
      .filter((reference) => selectedCandidateIds.has(reference.candidate_id))
      .map(searchCandidateToAnalysisReference)
      .filter((reference) => {
        const key = referenceStableKey(reference);
        if (existing.has(key)) return false;
        existing.add(key);
        return true;
      });
    setStagedAnalysisReferences((current) => {
      return [...current, ...selected];
    });
    setSearchStatus(t("status.addedToStaged", { count: selected.length }));
    setSearchStatusError(false);
    setSearchAnalysisQueued(false);
    showSearchStep(3);
  }

  function startNewSearchFlow() {
    const shouldContinue = window.confirm(t("search.newFlowConfirm"));
    if (!shouldContinue) return;
    resetSearchTask();
    setSearchStep(1);
    navigate("search");
  }

  function startNewStandaloneSearchTask() {
    const shouldContinue = !standaloneSearchLoading && !standaloneHasSearchResult && !standaloneActiveSearchHistoryId
      ? true
      : window.confirm(t("standaloneSearch.newTaskConfirm"));
    if (!shouldContinue) return;
    resetStandaloneSearchTask();
    setStandaloneSearchStatus(t("standaloneSearch.newTaskReady"));
    navigate("standaloneSearch");
  }

  function resetStandaloneSearchTask() {
    standaloneSearchRunId.current += 1;
    setStandaloneSearchLoading(false);
    setStandaloneSearchForm((current) => ({
      ...current,
      query: "",
      year: "",
    }));
    setStandaloneSearchCandidateReferences([]);
    setStandaloneSelectedCandidateIds(new Set());
    setStandaloneCandidatePayload({ rejected_count: 0, errors: {} });
    setStandaloneHasSearchResult(false);
    setStandaloneActiveSearchHistoryId("");
    setStandaloneSearchStatus("");
    setStandaloneSearchStatusError(false);
  }

  function startNewNoveltyTask() {
    const shouldContinue = !noveltyRunning && !noveltyResult && !noveltyActiveHistoryId
      ? true
      : window.confirm(t("novelty.newTaskConfirm"));
    if (!shouldContinue) return;
    resetNoveltyTask();
    setNoveltyStatus(t("novelty.newTaskReady"));
    navigate("novelty");
  }

  function resetNoveltyTask() {
    noveltyRunId.current += 1;
    setNoveltyRunning(false);
    setNoveltyForm((current) => ({
      ...current,
      innovationText: "",
      year: "",
    }));
    setNoveltyResult(null);
    setNoveltyActiveHistoryId("");
    setNoveltyStatus("");
    setNoveltyStatusError(false);
  }

  function sendStandaloneSearchToFlow() {
    setSearchForm({ ...standaloneSearchForm });
    setSearchCandidateReferences(standaloneSearchCandidateReferences.map((reference) => ({ ...reference })));
    setSelectedCandidateIds(new Set(standaloneSelectedCandidateIds));
    setCandidatePayload({ ...standaloneCandidatePayload });
    setHasSearchResult(standaloneHasSearchResult);
    setActiveSearchHistoryId(standaloneActiveSearchHistoryId);
    setSearchStatus(standaloneSearchStatus || (standaloneHasSearchResult ? t("history.continueSearchLoaded", { count: standaloneSearchCandidateReferences.length }) : ""));
    setSearchStatusError(standaloneSearchStatusError);
    setSearchLoading(false);
    setSearchAnalysisQueued(false);
    setStagedAnalysisReferences([]);
    updateResult("search", createEmptyAnalysisResult());
    setSearchStep(standaloneHasSearchResult ? 2 : 1);
    navigate("search");
  }

  function resetSearchTask() {
    searchRunId.current += 1;
    analysisRunIds.current.search += 1;
    setSearchLoading(false);
    setSearchForm((current) => ({
      ...current,
      query: "",
      year: "",
    }));
    setSearchCandidateReferences([]);
    setSelectedCandidateIds(new Set());
    setStagedAnalysisReferences([]);
    setCandidatePayload({ rejected_count: 0, errors: {} });
    setHasSearchResult(false);
    setSearchAnalysisQueued(false);
    setActiveSearchHistoryId("");
    setAnalysisRunning((current) => ({ ...current, search: false }));
    setSearchStatus("");
    setSearchStatusError(false);
    updateResult("search", createEmptyAnalysisResult());
  }

  function startNewDirectTask() {
    const shouldContinue = !analysisRunning.direct && !hasDirectAnalysisResult
      ? true
      : window.confirm(t("direct.newTaskConfirm"));
    if (!shouldContinue) return;
    analysisRunIds.current.direct += 1;
    setAnalysisRunning((current) => ({ ...current, direct: false }));
    setActiveAnalysisSource("direct");
    setAnalysisMode("idle");
    setAnalysisError("");
    setDoiInput("");
    setSelectedPdfFiles([]);
    setDirectStatus(t("direct.newTaskReady"));
    setDirectStatusError(false);
    updateResult("direct", createEmptyAnalysisResult());
    setExportFormat("md");
    navigate("direct");
  }

  function removeStagedAnalysisReference(index) {
    setStagedAnalysisReferences((current) => {
      if (index < 0 || index >= current.length) return current;
      const removed = current[index];
      const removedKey = referenceStableKey(removed);
      setSelectedCandidateIds((ids) => {
        const next = new Set(ids);
        searchCandidateReferences.forEach((candidate) => {
          if (referenceStableKey(candidate) === removedKey) next.delete(candidate.candidate_id);
        });
        return next;
      });
      const nextReferences = current.filter((_, itemIndex) => itemIndex !== index);
      setSearchStatus(nextReferences.length ? t("status.stagedRemaining", { count: nextReferences.length }) : t("status.stagedCleared"));
      setSearchStatusError(false);
      return nextReferences;
    });
  }

  function handleExport() {
    const setStatus = activeAnalysisSource === "search" ? setSearchStatus : setDirectStatus;
    const setStatusError = activeAnalysisSource === "search" ? setSearchStatusError : setDirectStatusError;
    exportAnalysisDocument({
      format: exportFormat,
      rows: activeResult.rows,
      summary: activeResult.summary,
      topic: t("analysis.reportTitle"),
      onStatus: (message, isError = false) => {
        setStatus(message);
        setStatusError(isError);
      },
      t,
    });
  }

  function handleTopbarNavigate(view) {
    if (view === "direct" && (analysisRunning.direct || analysisResultsBySource.direct.rows.length || analysisResultsBySource.direct.summary)) {
      setActiveAnalysisSource("direct");
      navigate("results");
      return;
    }
    if (view === "search" && searchStep === 4 && analysisRunning.search) {
      setActiveAnalysisSource("search");
      navigate("search");
      return;
    }
    if (view === "search" && searchStep === 4 && hasSearchAnalysisResult) {
      setActiveAnalysisSource("search");
      navigate("results");
      return;
    }
    navigate(view);
  }

  const topbarCurrentView = currentView === "results" ? activeAnalysisSource : currentView;
  const showStartNewSearchFlow = searchStep > 1 && Boolean(
    activeSearchHistoryId ||
    searchLoading ||
    hasSearchResult ||
    stagedAnalysisReferences.length ||
    searchAnalysisQueued
  );

  return (
    <>
      <Topbar
        currentView={topbarCurrentView}
        language={language}
        onLanguageChange={handleLanguageChange}
        onNavigate={handleTopbarNavigate}
        t={t}
      />
      <main className={`shell shell--${currentView}`}>
        {currentView === "home" ? <HomeView onNavigate={navigate} t={t} /> : null}
        {currentView === "direct" ? (
          <DirectAnalysisView
            status={directStatus}
            statusError={directStatusError}
            doiInput={doiInput}
            isRunning={analysisRunning.direct}
            selectedPdfFiles={selectedPdfFiles}
            pdfSummary={pdfSummary}
            linkSummary={linkSummary}
            onDoiInputChange={setDoiInput}
            onFilesSelected={handleFilesSelected}
            onRemoveFile={handleRemoveSelectedFile}
            onClearFiles={() => setSelectedPdfFiles([])}
            onAnalyze={() => submitLiteratureAnalysisFromSource("direct")}
            onStartNewTask={startNewDirectTask}
            showStartNewTask={analysisRunning.direct || hasDirectAnalysisResult}
            hasResult={hasDirectAnalysisResult}
            t={t}
          />
        ) : null}
        {currentView === "standaloneSearch" ? (
          <StandaloneSearchView
            searchForm={standaloneSearchForm}
            searchStatus={standaloneSearchStatus}
            searchStatusError={standaloneSearchStatusError}
            searchLoading={standaloneSearchLoading}
            candidateMeta={standaloneCandidateMeta}
            candidates={standaloneSearchCandidateReferences}
            selectedCandidateIds={standaloneSelectedCandidateIds}
            showStartNewTask={Boolean(standaloneActiveSearchHistoryId || standaloneSearchLoading || standaloneHasSearchResult)}
            hasSearchResult={standaloneHasSearchResult}
            onSearchFormChange={handleStandaloneSearchFormChange}
            onToggleSource={handleStandaloneToggleSource}
            onSubmitSearch={submitStandaloneLiteratureSearch}
            onToggleCandidate={(candidateId, isSelected) => {
              setStandaloneSelectedCandidateIds((current) => {
                const next = new Set(current);
                if (isSelected) next.add(candidateId);
                else next.delete(candidateId);
                return next;
              });
            }}
            onStartNewTask={startNewStandaloneSearchTask}
            onGoToSearchFlow={sendStandaloneSearchToFlow}
            t={t}
            language={language}
          />
        ) : null}
        {currentView === "novelty" ? (
          <NoveltyCheckView
            form={noveltyForm}
            status={noveltyStatus}
            statusError={noveltyStatusError}
            isRunning={noveltyRunning}
            result={noveltyResult}
            hasResult={Boolean(noveltyResult)}
            onFormChange={handleNoveltyFormChange}
            onToggleSource={handleNoveltyToggleSource}
            onSubmit={submitNoveltyCheck}
            onStartNewTask={startNewNoveltyTask}
            t={t}
          />
        ) : null}
        {currentView === "search" ? (
          <SearchFlowView
            activeStep={searchStep}
            searchForm={searchForm}
            searchStatus={searchStatus}
            searchStatusError={searchStatusError}
            searchLoading={searchLoading}
            candidateMeta={candidateMeta}
            candidates={searchCandidateReferences}
            selectedCandidateIds={selectedCandidateIds}
            stagedReferences={stagedAnalysisReferences}
            analysisRunning={analysisRunning.search}
            analysisQueued={searchAnalysisQueued}
            showStartNewFlow={showStartNewSearchFlow}
            hasAnalysisResult={hasSearchAnalysisResult}
            hasSearchResult={hasSearchResult}
            onStepChange={showSearchStep}
            onSearchFormChange={handleSearchFormChange}
            onToggleSource={handleToggleSource}
            onSubmitSearch={() => submitLiteratureSearch("search")}
            onToggleCandidate={toggleCandidate}
            onAddSelected={addSelectedSearchReferencesToAnalysis}
            onRemoveStaged={removeStagedAnalysisReference}
            onAnalyze={() => submitLiteratureAnalysisFromSource("search")}
            onStartNewFlow={startNewSearchFlow}
            onViewResults={() => {
              setActiveAnalysisSource("search");
              setSearchStep(4);
              navigate("results");
            }}
            t={t}
            language={language}
          />
        ) : null}
        {currentView === "results" ? (
          <ResultsView
            result={activeResult}
            mode={analysisMode}
            errorMessage={analysisError}
            exportFormat={exportFormat}
            exportStatusError={activeAnalysisSource === "search" ? searchStatusError : directStatusError}
            showSearchFlowStepper={activeAnalysisSource === "search"}
            searchStep={searchStep}
            onSearchStepChange={(step) => {
              if (step === 4 && analysisRunning.search) {
                setSearchStep(4);
                setActiveAnalysisSource("search");
                navigate("search");
                return;
              }
              if (step === 4 && hasSearchAnalysisResult) {
                setSearchStep(4);
                setActiveAnalysisSource("search");
                navigate("results");
                return;
              }
              showSearchStep(step);
            }}
            onExportFormatChange={setExportFormat}
            onExport={handleExport}
            onStartNewTask={activeAnalysisSource === "direct" ? startNewDirectTask : undefined}
            showStartNewTask={activeAnalysisSource === "direct" && (analysisRunning.direct || hasDirectAnalysisResult)}
            t={t}
          />
        ) : null}
        {currentView === "history" ? (
          <HistoryView
            entries={historyEntries}
            selectedEntry={selectedHistoryEntry}
            isLoading={historyLoading}
            errorMessage={historyError}
            onRefresh={() => loadHistory()}
            onSelectEntry={handleSelectHistoryEntry}
            onDeleteEntry={handleDeleteHistoryEntry}
            onResubmitEntry={handleResubmitHistoryEntry}
            onContinueEntry={handleContinueHistoryEntry}
            t={t}
          />
        ) : null}
      </main>
    </>
  );
}
