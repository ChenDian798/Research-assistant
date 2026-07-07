import { useEffect, useMemo, useRef, useState } from "react";
import CandidateList from "../components/CandidateList.jsx";
import LoadingState from "../components/LoadingState.jsx";

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
  ["computer", "search.mode.computer"],
  ["engineering", "search.mode.engineering"],
  ["society", "search.mode.society"],
];

export default function StandaloneSearchView({
  searchForm,
  searchStatus,
  searchStatusError,
  searchLoading,
  candidateMeta,
  candidates,
  selectedCandidateIds,
  showStartNewTask = false,
  hasSearchResult = false,
  onSearchFormChange,
  onToggleSource,
  onSubmitSearch,
  onToggleCandidate,
  onStartNewTask,
  onGoToSearchFlow,
  t,
  language = "zh",
}) {
  const [settingsMenuOpen, setSettingsMenuOpen] = useState(false);
  const [settingsBranch, setSettingsBranch] = useState("");
  const settingsMenuRef = useRef(null);
  const queryTextareaRef = useRef(null);
  const isEnglish = language === "en";
  const sourceLabel = (label) => label.startsWith("search.") ? t(label) : label;
  const visibleSearchStatus = searchStatus || t("standaloneSearch.statusIdle");
  const sourceConnectionText = isEnglish
    ? { connected: "On", connect: "Connect" }
    : { connected: "已连接", connect: "连接" };
  const modeChoiceText = isEnglish
    ? { selected: "Selected", select: "Select" }
    : { selected: "已选", select: "选择" };
  const selectedSourceOptions = useMemo(
    () => sourceOptions.filter(([value]) => searchForm.sources.includes(value)),
    [searchForm.sources],
  );
  const sourceButtonLabel = selectedSourceOptions.length
    ? selectedSourceOptions.map(([, label]) => sourceLabel(label)).join(" / ")
    : t("search.sources");
  const selectedModeOption = searchModeOptions.find(([value]) => value === searchForm.searchMode) || searchModeOptions[0];
  const modeButtonLabel = t(selectedModeOption[1]);
  const filterLabel = isEnglish ? "Filter" : "筛选";
  const anyYearLabel = isEnglish ? "Any year" : "任意年份";
  const perSourceLabel = isEnglish ? "each" : "每源";
  const filterSummary = `${searchForm.year || anyYearLabel} / ${searchForm.limit || "-"} ${perSourceLabel}`;

  const shouldShowResultsPanel = searchLoading || hasSearchResult || searchStatusError;
  const hasQueryInput = Boolean(searchForm.query.trim());
  const submitButtonState = searchLoading ? "is-loading" : (hasQueryInput ? "is-ready" : "is-empty");

  useEffect(() => {
    const textarea = queryTextareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [searchForm.query]);

  const toggleSettingsMenu = () => {
    setSettingsMenuOpen((isOpen) => {
      if (isOpen) setSettingsBranch("");
      return !isOpen;
    });
  };

  useEffect(() => {
    if (!settingsMenuOpen) return undefined;
    const handlePointerDown = (event) => {
      if (settingsMenuRef.current && !settingsMenuRef.current.contains(event.target)) {
        setSettingsMenuOpen(false);
        setSettingsBranch("");
      }
    };
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setSettingsMenuOpen(false);
        setSettingsBranch("");
      }
    };
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [settingsMenuOpen]);

  return (
    <section id="standaloneSearchView" className="app-view search-flow-view standalone-search-view is-active">
      <div className="view-header standalone-search-hero split-entry-hero split-entry-hero--standalone">
        <div className="split-entry-hero__copy">
          <p className="eyebrow">Search</p>
          <h1>{t("standaloneSearch.title")}</h1>
          <p className="view-lead">{t("standaloneSearch.lead")}</p>
        </div>
        <div className="split-entry-hero__stage">
          <div className="search-header-actions">
            <span className={`analysis-status ${searchStatusError ? "error" : ""}`} role="status" aria-live="polite">
              {visibleSearchStatus}
            </span>
            {showStartNewTask ? (
              <button className="primary-button search-new-flow-button" type="button" onClick={onStartNewTask}>
                {t("standaloneSearch.startNewTask")}
              </button>
            ) : null}
          </div>
          <div className="split-entry-hero__dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
        </div>
      </div>

      <section className="panel standalone-search-panel standalone-search-entry-panel">
        <div className="panel-body">
          <div className="search-composer">
            <label className="search-field search-field-wide search-query-field">
              <span>{t("search.query")}</span>
              <textarea
                ref={queryTextareaRef}
                rows="1"
                value={searchForm.query}
                placeholder="low-resource medical image segmentation foundation model"
                onChange={(event) => onSearchFormChange("query", event.target.value)}
              />
            </label>
            <div className="search-composer-toolbar">
              <div className="search-toolbar-left">
                <div className={`standalone-options-menu ${settingsMenuOpen ? "is-open" : ""}`} ref={settingsMenuRef}>
                  <button
                    className="standalone-options-trigger"
                    type="button"
                    aria-haspopup="dialog"
                    aria-expanded={settingsMenuOpen}
                    aria-label="Search settings"
                    title="Search settings"
                    onClick={toggleSettingsMenu}
                  >
                      <span className="standalone-options-plus" aria-hidden="true" />
                  </button>
                  {settingsMenuOpen ? (
                    <div className={`standalone-options-popover ${settingsBranch ? "is-branch" : "is-root"}`} role="dialog" aria-label="Search settings">
                      {!settingsBranch ? (
                        <div className="standalone-options-branches">
                          <button className="standalone-options-branch" type="button" onClick={() => setSettingsBranch("source")}>
                            <span>{t("search.sources")}</span>
                            <strong>{sourceButtonLabel}</strong>
                            <i aria-hidden="true" />
                          </button>
                          <button className="standalone-options-branch" type="button" onClick={() => setSettingsBranch("mode")}>
                            <span>{t("search.mode")}</span>
                            <strong>{modeButtonLabel}</strong>
                            <i aria-hidden="true" />
                          </button>
                          <button className="standalone-options-branch" type="button" onClick={() => setSettingsBranch("filter")}>
                            <span>{filterLabel}</span>
                            <strong>{filterSummary}</strong>
                            <i aria-hidden="true" />
                          </button>
                        </div>
                      ) : (
                        <>
                          <button className="standalone-options-back" type="button" onClick={() => setSettingsBranch("")}>
                            <span aria-hidden="true" />
                            {settingsBranch === "source" ? t("search.sources") : settingsBranch === "mode" ? t("search.mode") : filterLabel}
                          </button>
                          {settingsBranch === "source" ? (
                            <div className="standalone-options-section">
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
                          {settingsBranch === "mode" ? (
                            <div className="standalone-options-section">
                              {searchModeOptions.map(([value, label], index) => {
                                const checked = searchForm.searchMode === value;
                                const readableLabel = t(label);
                                return (
                                  <label className="source-app-option mode-app-option" role="menuitemradio" aria-checked={checked} key={value}>
                                    <input
                                      type="radio"
                                      name="standaloneSearchMode"
                                      checked={checked}
                                      onChange={() => onSearchFormChange("searchMode", value)}
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
                          {settingsBranch === "filter" ? (
                            <div className="standalone-options-section">
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
                        </>
                      )}
                    </div>
                  ) : null}
                </div>
                <span className="search-composer-summary">{sourceButtonLabel}</span>
                <span className="search-composer-sources">{modeButtonLabel} / {filterSummary}</span>
              </div>
              <button
                className={`search-submit-button search-submit-button--icon ${submitButtonState}`}
                type="button"
                disabled={searchLoading}
                aria-label={searchLoading ? t("search.searching") : t("search.submit")}
                title={searchLoading ? t("search.searching") : t("search.submit")}
                onClick={onSubmitSearch}
              >
                <span className={searchLoading ? "search-submit-stop" : "search-submit-arrow"} aria-hidden="true" />
                <span className="sr-only">{searchLoading ? t("search.searching") : t("search.submit")}</span>
              </button>
            </div>
          </div>
          <div className="advanced-options">
            <label><input type="checkbox" checked={searchForm.includeNeedsReview} onChange={(event) => onSearchFormChange("includeNeedsReview", event.target.checked)} /> {t("search.showReview")}</label>
            <label><input type="checkbox" checked={searchForm.appendAnnotationRecord} onChange={(event) => onSearchFormChange("appendAnnotationRecord", event.target.checked)} /> {t("search.writeRecord")}</label>
          </div>
        </div>
      </section>

      {shouldShowResultsPanel ? (
        <section className="panel standalone-search-panel standalone-search-results" aria-live="polite">
          <div className="panel-header candidate-panel-heading">
            <div>
              <h2>{t("standaloneSearch.resultsTitle")}</h2>
              {!searchLoading ? <p className="candidate-meta">{candidateMeta}</p> : null}
            </div>
            {hasSearchResult && !searchLoading ? (
              <button className="ghost-button" type="button" onClick={onGoToSearchFlow}>
                {t("standaloneSearch.sendToFlow")}
              </button>
            ) : null}
          </div>
          <div className="panel-body">
            {searchLoading ? (
              <LoadingState title={t("search.loadingTitle")} message={t("search.loadingMessage")} />
            ) : hasSearchResult ? (
              <CandidateList candidates={candidates} selectedIds={selectedCandidateIds} meta="" onToggle={onToggleCandidate} t={t} />
            ) : (
              <div className="candidate-list">
                <div className="candidate-empty">{t("standaloneSearch.emptyResults")}</div>
              </div>
            )}
          </div>
        </section>
      ) : null}
    </section>
  );
}
