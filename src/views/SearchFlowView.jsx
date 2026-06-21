import { useEffect, useMemo, useRef, useState } from "react";
import CandidateList from "../components/CandidateList.jsx";
import LoadingState from "../components/LoadingState.jsx";
import StagedReferenceList from "../components/StagedReferenceList.jsx";
import Stepper from "../components/Stepper.jsx";

const sourceOptions = [
  ["arxiv", "arXiv"],
  ["pubmed", "PubMed"],
  ["semantic", "Semantic Scholar"],
  ["crossref", "Crossref"],
  ["openalex", "OpenAlex"],
  ["cnki", "search.cnki"],
];

const searchModeOptions = [
  ["auto", "search.mode.auto"],
  ["biomedical", "search.mode.biomedical"],
  ["society", "search.mode.society"],
];

export default function SearchFlowView({
  activeStep,
  searchForm,
  searchStatus,
  searchStatusError,
  searchLoading,
  candidateMeta,
  candidates,
  selectedCandidateIds,
  stagedReferences,
  analysisRunning,
  analysisQueued = false,
  showStartNewFlow = false,
  hasAnalysisResult = false,
  hasSearchResult,
  onStepChange,
  onSearchFormChange,
  onToggleSource,
  onSubmitSearch,
  onToggleCandidate,
  onAddSelected,
  onRemoveStaged,
  onAnalyze,
  onViewResults,
  onStartNewFlow,
  t,
  language = "zh",
}) {
  const [sourceMenuOpen, setSourceMenuOpen] = useState(false);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const [filterMenuOpen, setFilterMenuOpen] = useState(false);
  const sourceMenuRef = useRef(null);
  const modeMenuRef = useRef(null);
  const filterMenuRef = useRef(null);
  const isEnglish = language === "en";
  const sourceLabel = (label) => label.startsWith("search.") ? t(label) : label;
  const visibleSearchStatus = searchStatus || t("status.waitingSearch");
  const sourceConnectionText = isEnglish
    ? { connected: "On", connect: "Connect" }
    : { connected: "\u5df2\u8fde\u63a5", connect: "\u8fde\u63a5" };
  const modeChoiceText = isEnglish
    ? { selected: "Selected", select: "Select" }
    : { selected: "\u5df2\u9009", select: "\u9009\u62e9" };
  const selectedSourceOptions = useMemo(
    () => sourceOptions.filter(([value]) => searchForm.sources.includes(value)),
    [searchForm.sources],
  );
  const sourceButtonLabel = selectedSourceOptions.length
    ? selectedSourceOptions.map(([, label]) => sourceLabel(label)).join(" / ")
    : t("search.sources");
  const selectedModeOption = searchModeOptions.find(([value]) => value === searchForm.searchMode) || searchModeOptions[0];
  const modeButtonLabel = t(selectedModeOption[1]);
  const filterLabel = isEnglish ? "Filter" : "\u7b5b\u9009";
  const anyYearLabel = isEnglish ? "Any year" : "\u4efb\u610f\u5e74\u4efd";
  const perSourceLabel = isEnglish ? "each" : "\u6bcf\u6e90";
  const filterSummary = `${searchForm.year || anyYearLabel} / ${searchForm.limit || "-"} ${perSourceLabel}`;
  const analysisActionLabel = analysisRunning
    ? t("search.viewAnalysisProgress")
    : (analysisQueued ? t("search.analysisSubmitted") : (hasAnalysisResult ? t("search.viewAnalysisResult") : t("search.nextAnalyze")));
  const handleStepChange = (step) => {
    if (step === 4 && (analysisRunning || hasAnalysisResult)) {
      onViewResults();
      return;
    }
    onStepChange(step);
  };

  useEffect(() => {
    if (!sourceMenuOpen && !modeMenuOpen && !filterMenuOpen) return undefined;
    const handlePointerDown = (event) => {
      if (sourceMenuRef.current && !sourceMenuRef.current.contains(event.target)) {
        setSourceMenuOpen(false);
      }
      if (modeMenuRef.current && !modeMenuRef.current.contains(event.target)) {
        setModeMenuOpen(false);
      }
      if (filterMenuRef.current && !filterMenuRef.current.contains(event.target)) {
        setFilterMenuOpen(false);
      }
    };
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setSourceMenuOpen(false);
        setModeMenuOpen(false);
        setFilterMenuOpen(false);
      }
    };
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [sourceMenuOpen, modeMenuOpen, filterMenuOpen]);

  return (
    <section id="searchView" className="app-view search-flow-view is-active">
      <div className="view-header">
        <div>
          <p className="eyebrow">Search Flow</p>
          <h1>{t("search.title")}</h1>
          <p className="view-lead">{t("search.lead")}</p>
        </div>
        <div className="search-header-actions">
          <span className={`analysis-status ${searchStatusError ? "error" : ""}`} role="status" aria-live="polite">{visibleSearchStatus}</span>
          {showStartNewFlow ? (
            <button className="primary-button search-new-flow-button" type="button" onClick={onStartNewFlow}>{t("search.startNewFlow")}</button>
          ) : null}
        </div>
      </div>

      <Stepper activeStep={activeStep} onStepChange={handleStepChange} t={t} />

      {activeStep === 1 ? (
        <section className="panel flow-step is-active">
          <div className="panel-header">
            <div>
              <h2>{t("search.step1Title")}</h2>
              <p>{t("search.step1Body")}</p>
            </div>
          </div>
          <div className="panel-body">
            <div className="search-composer">
              <label className="search-field search-field-wide search-query-field">
                <span>{t("search.query")}</span>
                <textarea rows="4" value={searchForm.query} placeholder="low-resource medical image segmentation foundation model" onChange={(event) => onSearchFormChange("query", event.target.value)} />
              </label>
              <div className="search-composer-toolbar">
                <div className="search-toolbar-left">
                  <div className={`source-app-menu ${sourceMenuOpen ? "is-open" : ""}`} ref={sourceMenuRef}>
                    <button
                      className="source-app-trigger"
                      type="button"
                      aria-haspopup="menu"
                      aria-expanded={sourceMenuOpen}
                      onClick={() => {
                        setModeMenuOpen(false);
                        setFilterMenuOpen(false);
                        setSourceMenuOpen((isOpen) => !isOpen);
                      }}
                    >
                      <span className="source-app-grid-icon" aria-hidden="true">
                        <i />
                        <i />
                        <i />
                        <i />
                      </span>
                      <span className="source-app-trigger-label">{sourceButtonLabel}</span>
                      <span className="source-app-chevron" aria-hidden="true" />
                    </button>
                    {sourceMenuOpen ? (
                      <div className="source-app-popover" role="menu" aria-label={t("search.sources")}>
                        <div className="source-app-popover-title">{t("search.sources")}</div>
                        {sourceOptions.map(([value, label]) => {
                          const checked = searchForm.sources.includes(value);
                          const readableLabel = sourceLabel(label);
                          return (
                            <label className="source-app-option" role="menuitemcheckbox" aria-checked={checked} key={value}>
                              <input type="checkbox" checked={checked} onChange={(event) => onToggleSource(value, event.target.checked)} />
                              <span className={`source-app-mark source-app-mark--${value}`} aria-hidden="true">
                                {readableLabel.slice(0, 1)}
                              </span>
                              <span className="source-app-name">{readableLabel}</span>
                              <span className="source-app-connect">{checked ? sourceConnectionText.connected : sourceConnectionText.connect}</span>
                            </label>
                          );
                        })}
                      </div>
                    ) : null}
                  </div>
                  <div className={`source-app-menu mode-app-menu ${modeMenuOpen ? "is-open" : ""}`} ref={modeMenuRef}>
                    <button
                      className="source-app-trigger mode-app-trigger"
                      type="button"
                      aria-haspopup="menu"
                      aria-expanded={modeMenuOpen}
                      onClick={() => {
                        setSourceMenuOpen(false);
                        setFilterMenuOpen(false);
                        setModeMenuOpen((isOpen) => !isOpen);
                      }}
                    >
                      <span className={`mode-app-icon mode-app-mark--${selectedModeOption[0]}`} aria-hidden="true">{modeButtonLabel.slice(0, 1)}</span>
                      <span className="source-app-trigger-label">{modeButtonLabel}</span>
                      <span className="source-app-chevron" aria-hidden="true" />
                    </button>
                    {modeMenuOpen ? (
                      <div className="source-app-popover mode-app-popover" role="menu" aria-label={t("search.mode")}>
                        <div className="source-app-popover-title">{t("search.mode")}</div>
                        {searchModeOptions.map(([value, label], index) => {
                          const checked = searchForm.searchMode === value;
                          const readableLabel = t(label);
                          return (
                            <label className="source-app-option mode-app-option" role="menuitemradio" aria-checked={checked} key={value}>
                              <input
                                type="radio"
                                name="searchMode"
                                checked={checked}
                                onChange={() => {
                                  onSearchFormChange("searchMode", value);
                                  setModeMenuOpen(false);
                                }}
                              />
                              <span className={`source-app-mark mode-app-mark mode-app-mark--${value}`} aria-hidden="true">
                                {readableLabel.slice(0, 1) || index + 1}
                              </span>
                              <span className="source-app-name">{readableLabel}</span>
                              <span className="source-app-connect">{checked ? modeChoiceText.selected : modeChoiceText.select}</span>
                            </label>
                          );
                        })}
                      </div>
                    ) : null}
                  </div>
                  <div className={`source-app-menu filter-app-menu ${filterMenuOpen ? "is-open" : ""}`} ref={filterMenuRef}>
                    <button
                      className="source-app-trigger filter-app-trigger"
                      type="button"
                      aria-haspopup="dialog"
                      aria-expanded={filterMenuOpen}
                      onClick={() => {
                        setSourceMenuOpen(false);
                        setModeMenuOpen(false);
                        setFilterMenuOpen((isOpen) => !isOpen);
                      }}
                    >
                      <span className="filter-app-icon" aria-hidden="true" />
                      <span className="source-app-trigger-label">{filterLabel}</span>
                      <span className="filter-app-summary">{filterSummary}</span>
                      <span className="source-app-chevron" aria-hidden="true" />
                    </button>
                    {filterMenuOpen ? (
                      <div className="source-app-popover filter-app-popover" role="dialog" aria-label={filterLabel}>
                        <div className="source-app-popover-title">{filterLabel}</div>
                        <label className="search-mini-field filter-mini-field">
                          <span>{t("search.year")}</span>
                          <input type="text" value={searchForm.year} placeholder="2022-2026" onChange={(event) => onSearchFormChange("year", event.target.value)} />
                        </label>
                        <label className="search-mini-field search-mini-field--limit filter-mini-field">
                          <span>{t("search.limit")}</span>
                          <input type="number" min="1" max="50" value={searchForm.limit} onChange={(event) => onSearchFormChange("limit", event.target.value)} />
                        </label>
                      </div>
                    ) : null}
                  </div>
                </div>
                <button className="search-submit-button" type="button" disabled={searchLoading} onClick={onSubmitSearch} aria-label={searchLoading ? t("search.searching") : t("search.submit")}>
                  <span>{searchLoading ? t("search.searching") : t("search.submit")}</span>
                </button>
              </div>
            </div>
            <div className="advanced-options">
              <label><input type="checkbox" checked={searchForm.includeNeedsReview} onChange={(event) => onSearchFormChange("includeNeedsReview", event.target.checked)} /> {t("search.showReview")}</label>
              <label><input type="checkbox" checked={searchForm.appendAnnotationRecord} onChange={(event) => onSearchFormChange("appendAnnotationRecord", event.target.checked)} /> {t("search.writeRecord")}</label>
            </div>
          </div>
        </section>
      ) : null}

      {activeStep === 2 ? (
        <section id="searchCandidatePanel" className="panel flow-step is-active" aria-live="polite">
          <div className="panel-header candidate-panel-heading">
            <div>
              <h2>{t("search.step2Title")}</h2>
              {!searchLoading ? <p className="candidate-meta">{candidateMeta}</p> : null}
            </div>
          </div>
          <div className="panel-body">
            {searchLoading ? (
              <LoadingState title={t("search.loadingTitle")} message={t("search.loadingMessage")} />
            ) : hasSearchResult ? (
              <CandidateList candidates={candidates} selectedIds={selectedCandidateIds} meta="" onToggle={onToggleCandidate} t={t} />
            ) : (
              <div className="candidate-list">
                <div className="candidate-empty">{t("search.emptyAfterSearch")}</div>
              </div>
            )}
            <div className="form-actions">
              <button className="ghost-button" type="button" onClick={() => onStepChange(1)}>{t("search.backSearch")}</button>
              <button className="primary-button" type="button" disabled={!selectedCandidateIds.size} onClick={onAddSelected}>{t("search.addToList")}</button>
            </div>
          </div>
        </section>
      ) : null}

      {activeStep === 3 ? (
        <section className="panel flow-step is-active">
          <div className="panel-header">
            <div>
              <h2>{t("search.step3Title")}</h2>
              <p>{t("search.step3Body")}</p>
            </div>
          </div>
          <div className="panel-body">
            <StagedReferenceList references={stagedReferences} onRemove={onRemoveStaged} t={t} />
            <div className="form-actions">
              <button className="ghost-button" type="button" onClick={() => onStepChange(2)}>{t("search.backResults")}</button>
              <button
                className="primary-button"
                type="button"
                onClick={() => {
                  if (analysisQueued) {
                    onStepChange(4);
                    return;
                  }
                  if (analysisRunning || hasAnalysisResult) {
                    onViewResults();
                    return;
                  }
                  onStepChange(4);
                  onAnalyze();
                }}
              >
                {analysisActionLabel}
              </button>
            </div>
          </div>
        </section>
      ) : null}

      {activeStep === 4 ? (
        <section className="panel flow-step is-active">
          <div className="panel-header">
            <div>
              <h2>{t("search.step4Title")}</h2>
              <p>{t("search.step4Body")}</p>
            </div>
          </div>
          <div className="panel-body">
            {analysisRunning ? (
              <LoadingState
                title={t("table.loadingTitle")}
                message={t("table.loadingWithCount", { count: stagedReferences.length })}
              />
            ) : hasAnalysisResult ? (
              <div className="empty-state-box">
                <strong>{t("search.analysisCompleteTitle")}</strong>
                <span>{t("search.analysisCompleteBody")}</span>
              </div>
            ) : analysisQueued ? (
              <LoadingState
                title={t("search.analysisQueuedTitle")}
                message={t("search.analysisQueuedBody")}
              />
            ) : (
              <div className="empty-state-box">
                <strong>{t("search.readyTitle")}</strong>
                <span>{t("search.readyBody")}</span>
              </div>
            )}
            <div className="form-actions">
              <button className="ghost-button" type="button" onClick={() => onStepChange(3)}>{t("search.backList")}</button>
              {!analysisRunning && !analysisQueued && !hasAnalysisResult ? (
                <button className="primary-button" type="button" disabled={!stagedReferences.length} onClick={onAnalyze}>
                  {t("search.nextAnalyze")}
                </button>
              ) : null}
              {hasAnalysisResult ? (
                <button className="primary-button" type="button" onClick={onViewResults}>{t("search.viewAnalysisResult")}</button>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}
    </section>
  );
}
