import { useEffect, useRef, useState } from "react";
import LoadingState from "../components/LoadingState.jsx";
import ReferenceLink from "../components/ReferenceLink.jsx";

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

const overlapPriority = {
  high_overlap: 4,
  partial_overlap: 3,
  adjacent: 2,
  no_clear_overlap: 1,
};

function normalizedOverlapLevel(value) {
  return String(value || "adjacent").trim() || "adjacent";
}

function sortedByOverlap(comparisons) {
  return [...comparisons].sort((left, right) => {
    const leftLevel = normalizedOverlapLevel(left.overlap_level);
    const rightLevel = normalizedOverlapLevel(right.overlap_level);
    const levelDelta = (overlapPriority[rightLevel] || 0) - (overlapPriority[leftLevel] || 0);
    if (levelDelta) return levelDelta;
    return Number(right.overlap_score || 0) - Number(left.overlap_score || 0);
  });
}

function firstListItem(value) {
  return Array.isArray(value) && value.length ? value[0] : "";
}

export default function NoveltyCheckView({
  form,
  status,
  statusError,
  isRunning,
  result,
  hasResult,
  onFormChange,
  onToggleSource,
  onSubmit,
  onStartNewTask,
  t,
}) {
  const textareaRef = useRef(null);
  const settingsMenuRef = useRef(null);
  const [settingsMenuOpen, setSettingsMenuOpen] = useState(false);
  const [settingsBranch, setSettingsBranch] = useState("");
  const sourceLabel = (label) => label.startsWith("search.") ? t(label) : label;
  const visibleStatus = status || t("novelty.statusIdle");
  const selectedSources = sourceOptions.filter(([value]) => form.sources.includes(value));
  const sourceButtonLabel = selectedSources.length
    ? selectedSources.map(([, label]) => sourceLabel(label)).join(" / ")
    : t("search.sources");
  const modeLabel = t((searchModeOptions.find(([value]) => value === form.searchMode) || searchModeOptions[0])[1]);
  const filterLabel = t("search.filter");
  const filterSummary = form.year?.trim() || t("search.anyYear");
  const hasClaimInput = Boolean(form.innovationText.trim());
  const submitButtonState = isRunning ? "is-loading" : (hasClaimInput ? "is-ready" : "is-empty");
  const toggleSettingsMenu = () => {
    setSettingsMenuOpen((isOpen) => {
      if (isOpen) setSettingsBranch("");
      return !isOpen;
    });
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  }, [form.innovationText]);

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
    <section id="noveltyCheckView" className="app-view novelty-check-view is-active">
      <div className="view-header novelty-check-hero split-entry-hero">
        <div className="split-entry-hero__copy">
          <p className="eyebrow">Novelty</p>
          <h1>{t("novelty.title")}</h1>
          <p className="view-lead">{t("novelty.lead")}</p>
        </div>
        <div className="split-entry-hero__stage">
          <div className="search-header-actions">
            <span className={`analysis-status ${statusError ? "error" : ""}`} role="status" aria-live="polite">
              {visibleStatus}
            </span>
            {hasResult || isRunning ? (
              <button className="primary-button search-new-flow-button" type="button" onClick={onStartNewTask}>
                {t("novelty.startNewTask")}
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

      <section className="panel novelty-check-panel novelty-check-entry-panel">
        <div className="panel-body">
          <div className="search-composer novelty-composer">
            <label className="search-field search-field-wide search-query-field novelty-claim-field">
              <span>{t("novelty.claimLabel")}</span>
              <textarea
                ref={textareaRef}
                rows="2"
                value={form.innovationText}
                placeholder={t("novelty.claimPlaceholder")}
                onChange={(event) => onFormChange("innovationText", event.target.value)}
              />
            </label>

            <div className="search-composer-toolbar novelty-composer-toolbar">
              <div className="search-toolbar-left">
                <div className={`standalone-options-menu novelty-options-menu ${settingsMenuOpen ? "is-open" : ""}`} ref={settingsMenuRef}>
                  <button
                    className="standalone-options-trigger"
                    type="button"
                    aria-haspopup="dialog"
                    aria-expanded={settingsMenuOpen}
                    aria-label={t("novelty.searchSettings")}
                    title={t("novelty.searchSettings")}
                    onClick={toggleSettingsMenu}
                  >
                    <span className="standalone-options-plus" aria-hidden="true" />
                  </button>
                  {settingsMenuOpen ? (
                    <div className={`standalone-options-popover ${settingsBranch ? "is-branch" : "is-root"}`} role="dialog" aria-label={t("novelty.searchSettings")}>
                      {!settingsBranch ? (
                        <div className="standalone-options-branches">
                          <button className="standalone-options-branch" type="button" onClick={() => setSettingsBranch("source")}>
                            <span>{t("search.sources")}</span>
                            <strong>{sourceButtonLabel}</strong>
                            <i aria-hidden="true" />
                          </button>
                          <button className="standalone-options-branch" type="button" onClick={() => setSettingsBranch("mode")}>
                            <span>{t("search.mode")}</span>
                            <strong>{modeLabel}</strong>
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
                                const checked = form.sources.includes(value);
                                const readableLabel = sourceLabel(label);
                                return (
                                  <label className="source-app-option" role="menuitemcheckbox" aria-checked={checked} key={value}>
                                    <input type="checkbox" checked={checked} onChange={(event) => onToggleSource(value, event.target.checked)} />
                                    <span className={`source-app-mark source-app-mark--${value}`} aria-hidden="true">
                                      {readableLabel.slice(0, 1)}
                                    </span>
                                    <span className="source-app-name">{readableLabel}</span>
                                    <span className="source-app-connect">{checked ? t("search.connected") : t("search.connect")}</span>
                                  </label>
                                );
                              })}
                            </div>
                          ) : null}
                          {settingsBranch === "mode" ? (
                            <div className="standalone-options-section">
                              {searchModeOptions.map(([value, label], index) => {
                                const checked = form.searchMode === value;
                                const readableLabel = t(label);
                                return (
                                  <label className="source-app-option mode-app-option" role="menuitemradio" aria-checked={checked} key={value}>
                                    <input type="radio" name="noveltySearchMode" checked={checked} onChange={() => onFormChange("searchMode", value)} />
                                    <span className={`source-app-mark mode-app-mark mode-app-mark--${value}`} aria-hidden="true">{readableLabel.slice(0, 1) || index + 1}</span>
                                    <span className="source-app-name">{readableLabel}</span>
                                    <span className="source-app-connect">{checked ? t("search.selected") : t("search.select")}</span>
                                  </label>
                                );
                              })}
                            </div>
                          ) : null}
                          {settingsBranch === "filter" ? (
                            <div className="standalone-options-section">
                              <label className="search-mini-field filter-mini-field">
                                <span>{t("search.year")}</span>
                                <input type="text" value={form.year} placeholder="2022-2026" onChange={(event) => onFormChange("year", event.target.value)} />
                              </label>
                              <label className="novelty-inline-option novelty-inline-option--menu">
                                <input type="checkbox" checked={form.includeFilteredReferences} onChange={(event) => onFormChange("includeFilteredReferences", event.target.checked)} />
                                {t("novelty.includeFiltered")}
                              </label>
                            </div>
                          ) : null}
                        </>
                      )}
                    </div>
                  ) : null}
                </div>
                <span className="search-composer-summary novelty-composer-summary">{sourceButtonLabel}</span>
                <span className="search-composer-sources novelty-composer-sources">{modeLabel} / {filterSummary}</span>
              </div>
              <button
                className={`search-submit-button search-submit-button--icon novelty-submit-button ${submitButtonState}`}
                type="button"
                disabled={isRunning}
                aria-label={isRunning ? t("novelty.running") : t("novelty.submit")}
                title={isRunning ? t("novelty.running") : t("novelty.submit")}
                onClick={onSubmit}
              >
                <span className={isRunning ? "search-submit-stop" : "search-submit-arrow"} aria-hidden="true" />
                <span className="sr-only">{isRunning ? t("novelty.running") : t("novelty.submit")}</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {isRunning ? (
        <section className="panel novelty-check-panel" aria-live="polite">
          <LoadingState title={t("novelty.loadingTitle")} message={t("novelty.loadingBody")} />
        </section>
      ) : hasResult ? (
        <NoveltyReport result={result} t={t} />
      ) : null}
    </section>
  );
}

export function NoveltyReport({ result, t }) {
  const overall = result?.overall || {};
  const comparisons = Array.isArray(result?.comparisons) ? result.comparisons : [];
  const claims = Array.isArray(result?.innovation_claims) ? result.innovation_claims : [];
  const nextSteps = Array.isArray(result?.next_steps) ? result.next_steps : [];
  const closestPriorWork = Array.isArray(result?.closest_prior_work) ? result.closest_prior_work : [];
  const noveltyDimensions = result?.novelty_dimensions && typeof result.novelty_dimensions === "object" ? result.novelty_dimensions : {};
  const innovationProfile = result?.innovation_profile && typeof result.innovation_profile === "object" ? result.innovation_profile : {};
  const counts = result?.counts || {};
  const diagnostics = result?.diagnostics || result?.search?.diagnostics || {};
  const risk = overall.risk_level || "unknown";
  const sortedComparisons = sortedByOverlap(comparisons);
  const topComparisons = sortedComparisons.slice(0, 5);
  const remainingComparisons = sortedComparisons.slice(5);
  const actionSteps = nextSteps.slice(0, 3);

  return (
    <section className="panel novelty-check-panel novelty-report" aria-live="polite">
      <div className="novelty-report-header">
        <div>
          <span className={`novelty-risk novelty-risk--${risk}`}>{t(`novelty.risk.${risk}`)}</span>
          <h2>{t("novelty.reportTitle")}</h2>
          <p>{overall.assessment || t("novelty.noAssessment")}</p>
        </div>
        <div className="novelty-score-grid">
          <Metric label={t("novelty.metricHigh")} value={counts.high_overlap || 0} />
          <Metric label={t("novelty.metricPartial")} value={counts.partial_overlap || 0} />
          <Metric label={t("novelty.metricAssessed")} value={counts.total ?? comparisons.length} />
        </div>
      </div>

      <InnovationProfileSummary profile={innovationProfile} t={t} />

      <PriorityPriorWork rows={closestPriorWork} comparisons={sortedComparisons} t={t} />

      {actionSteps.length ? (
        <div className="novelty-section novelty-next-priority">
          <h3>{t("novelty.nextStepsTitle")}</h3>
          <ul className="novelty-next-list">
            {actionSteps.map((step, index) => <li key={`${step}-${index}`}>{step}</li>)}
          </ul>
        </div>
      ) : null}

      <NoveltyDimensionSummary dimensions={noveltyDimensions} t={t} />

      <div className="novelty-section">
        <h3>{t("novelty.topComparisonsTitle")}</h3>
        {topComparisons.length ? (
          <div className="novelty-comparison-list">
            {topComparisons.map((comparison, index) => (
              <NoveltyComparison
                comparison={comparison}
                t={t}
                defaultOpen={index < 3 && ["high_overlap", "partial_overlap"].includes(normalizedOverlapLevel(comparison.overlap_level))}
                key={`${comparison.reference_index}-${comparison.title}`}
              />
            ))}
          </div>
        ) : (
          <div className="empty-state">{t("novelty.noComparisons")}</div>
        )}
      </div>

      {remainingComparisons.length ? (
        <details className="novelty-secondary-section">
          <summary>{t("novelty.remainingComparisons", { count: remainingComparisons.length })}</summary>
          <div className="novelty-comparison-list">
            {remainingComparisons.map((comparison) => (
              <NoveltyComparison comparison={comparison} t={t} key={`${comparison.reference_index}-${comparison.title}`} />
            ))}
          </div>
        </details>
      ) : null}

      {claims.length ? (
        <details className="novelty-secondary-section">
          <summary>{t("novelty.systemUnderstandingTitle")}</summary>
          <div className="novelty-section">
            <ul className="novelty-pill-list">
              {claims.map((claim, index) => <li key={`${claim}-${index}`}>{claim}</li>)}
            </ul>
          </div>
        </details>
      ) : null}

      <details className="novelty-secondary-section">
        <summary>{t("novelty.diagnosticsFoldTitle")}</summary>
        <NoveltyDiagnostics diagnostics={diagnostics} result={result} t={t} />
        <ClosestPriorWork rows={closestPriorWork} t={t} />
      </details>

      {nextSteps.length > actionSteps.length ? (
        <div className="novelty-section">
          <h3>{t("novelty.additionalStepsTitle")}</h3>
          <ul className="novelty-next-list">
            {nextSteps.slice(actionSteps.length).map((step, index) => <li key={`${step}-${index}`}>{step}</li>)}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

function InnovationProfileSummary({ profile, t }) {
  const types = Array.isArray(profile?.innovation_types) ? profile.innovation_types : [];
  const focus = Array.isArray(profile?.domain_focus) ? profile.domain_focus : [];
  const domain = profile?.domain || "general";
  if (!types.length && !focus.length) return null;
  return (
    <div className="novelty-section novelty-profile-summary">
      <div className="novelty-profile-heading">
        <h3>{t("novelty.profileTitle")}</h3>
        <span>{t("novelty.profileDomain")}: {t(`novelty.domain.${domain}`)}</span>
      </div>
      {types.length ? (
        <div className="novelty-profile-grid">
          {types.map((row, index) => (
            <ProfileCard
              key={`${row.type}-${index}`}
              title={t(`novelty.profile.type.${row.type}`)}
              risk={row.risk || "unknown"}
              assessment={row.assessment}
              t={t}
            />
          ))}
        </div>
      ) : null}
      {focus.length ? (
        <details className="novelty-profile-focus">
          <summary>{t("novelty.profileFocusTitle")}</summary>
          <div className="novelty-profile-grid">
            {focus.map((row, index) => (
              <ProfileCard
                key={`${row.key}-${index}`}
                title={t(`novelty.profile.focus.${row.key}`)}
                risk={row.risk || "unknown"}
                assessment={row.assessment}
                t={t}
              />
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

function ProfileCard({ title, risk, assessment, t }) {
  return (
    <article className="novelty-profile-card">
      <span className={`novelty-risk novelty-risk--${risk}`}>{t(`novelty.risk.${risk}`)}</span>
      <strong>{title}</strong>
      {assessment ? <p>{assessment}</p> : null}
    </article>
  );
}

function NoveltyDiagnostics({ diagnostics, result, t }) {
  const sourceSummary = diagnostics?.source_summary || {};
  const sourceRows = Object.entries(sourceSummary);
  const pool = diagnostics?.candidate_pool || {};
  const llmAssessment = result?.llm_assessment || {};
  const llmWarnings = Array.isArray(llmAssessment?.warnings) ? llmAssessment.warnings : [];
  const warnings = [
    ...(Array.isArray(diagnostics?.warnings) ? diagnostics.warnings : []),
    ...llmWarnings,
  ].filter(Boolean);
  const queryRows = Array.isArray(diagnostics?.queries) ? diagnostics.queries.slice(0, 10) : [];
  const hasDiagnostics = sourceRows.length || warnings.length || queryRows.length || Object.keys(pool).length;
  if (!hasDiagnostics) {
    const sourceResults = result?.source_results || result?.search?.source_results || {};
    const errors = result?.errors || result?.search?.errors || {};
    const fallbackRows = Object.entries(sourceResults);
    if (!fallbackRows.length && !Object.keys(errors).length && !result?.raw_count) return null;
    return (
      <div className="novelty-section novelty-diagnostics">
        <h3>{t("novelty.diagnosticsTitle")}</h3>
        <div className="novelty-diagnostics-grid">
          <Metric label={t("novelty.rawCount")} value={result?.raw_count || 0} />
          <Metric label={t("novelty.keptCount")} value={countsValue(result?.counts)} />
          <Metric label={t("novelty.filteredCount")} value={Math.max(0, Number(result?.raw_count || 0) - countsValue(result?.counts))} />
        </div>
        {warnings.length ? (
          <ul className="novelty-warning-list">
            {warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}
          </ul>
        ) : null}
        <SourceSummary rows={fallbackRows.map(([source, returned]) => [source, { returned, kept: "", filtered: "", error: errors[source] || "" }])} t={t} />
      </div>
    );
  }
  return (
    <div className="novelty-section novelty-diagnostics">
      <h3>{t("novelty.diagnosticsTitle")}</h3>
      <div className="novelty-diagnostics-grid">
        <Metric label={t("novelty.rawCount")} value={pool.raw ?? result?.raw_count ?? 0} />
        <Metric label={t("novelty.keptCount")} value={pool.sent_to_overlap_assessment ?? countsValue(result?.counts)} />
        <Metric label={t("novelty.filteredCount")} value={pool.noise ?? 0} />
        <Metric label={t("novelty.weakCount")} value={pool.weak ?? 0} />
      </div>
      <SourceSummary rows={sourceRows} t={t} />
      {warnings.length ? (
        <ul className="novelty-warning-list">
          {warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}
        </ul>
      ) : null}
      {queryRows.length ? (
        <div className="novelty-query-table" role="table" aria-label={t("novelty.queryDiagnostics")}>
          <div className="novelty-query-row novelty-query-row--head" role="row">
            <span>{t("novelty.queryId")}</span>
            <span>{t("novelty.source")}</span>
            <span>{t("novelty.returned")}</span>
            <span>{t("novelty.kept")}</span>
            <span>{t("novelty.filtered")}</span>
          </div>
          {queryRows.map((query, index) => (
            <div className="novelty-query-row" role="row" key={`${query.query_id}-${query.source}-${index}`}>
              <span title={query.query || ""}>{query.query_id || query.purpose || "-"}</span>
              <span>{query.source || "-"}</span>
              <span>{query.returned ?? 0}</span>
              <span>{query.kept ?? 0}</span>
              <span>{query.filtered ?? 0}{query.error ? ` / ${query.error}` : ""}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function SourceSummary({ rows, t }) {
  if (!rows.length) return null;
  return (
    <div className="novelty-source-summary">
      {rows.map(([source, summary]) => (
        <div className="novelty-source-summary-item" key={source}>
          <strong>{source}</strong>
          <span>{t("novelty.sourceReturned")}: {summary?.returned ?? 0}</span>
          <span>{t("novelty.sourceKept")}: {summary?.kept ?? "-"}</span>
          <span>{t("novelty.sourceFiltered")}: {summary?.filtered ?? "-"}</span>
          {summary?.error ? <em>{summary.error}</em> : null}
        </div>
      ))}
    </div>
  );
}

function countsValue(counts) {
  if (!counts || typeof counts !== "object") return 0;
  return Number(counts.total || 0);
}

function NoveltyComparison({ comparison, t, defaultOpen = false }) {
  const points = Array.isArray(comparison.overlap_points) ? comparison.overlap_points : [];
  const differences = Array.isArray(comparison.difference_points) ? comparison.difference_points : [];
  const dimensions = comparison.dimension_overlap || {};
  const overlapLevel = normalizedOverlapLevel(comparison.overlap_level);
  return (
    <article className={`novelty-comparison novelty-comparison--${overlapLevel}`}>
      <details open={defaultOpen}>
        <summary className="novelty-comparison-summary">
          <span className={`novelty-overlap novelty-overlap--${overlapLevel}`}>
            {t(`novelty.overlap.${overlapLevel}`)}
          </span>
          <span className="novelty-comparison-title">
            <ReferenceLink title={comparison.title || t("candidate.untitled")} source={comparison.source || ""} t={t} />
          </span>
          <span className="candidate-byline">{[comparison.authors, comparison.year, comparison.source_label].filter(Boolean).join(" / ") || t("candidate.incompleteMeta")}</span>
          <span className="novelty-comparison-glance">
            {firstListItem(points) || comparison.evidence || t("novelty.noAssessment")}
          </span>
        </summary>
        <div className="novelty-comparison-body">
          <div className="novelty-comparison-main">
            <p className={`candidate-badge verification-${comparison.verification_status || "partial"}`}>
              {t("novelty.verificationStatus")}: {comparison.verification_status || "partial"}
            </p>
            {comparison.verification_note ? <p className="novelty-verification-note">{comparison.verification_note}</p> : null}
            {comparison.evidence ? <p className="novelty-evidence"><strong>{t("novelty.evidenceTitle")}</strong>{comparison.evidence}</p> : null}
          </div>
          <div className="novelty-comparison-detail">
            <DimensionList dimensions={dimensions} t={t} />
            <DetailList title={t("novelty.overlapPoints")} items={points} />
            <DetailList title={t("novelty.differencePoints")} items={differences} />
            {comparison.recommendation ? (
              <p className="novelty-recommendation"><strong>{t("novelty.recommendation")}</strong>{comparison.recommendation}</p>
            ) : null}
          </div>
        </div>
      </details>
    </article>
  );
}

function NoveltyDimensionSummary({ dimensions, t }) {
  const rows = [
    ["method_novelty", "novelty.dimension.methodNovelty"],
    ["application_novelty", "novelty.dimension.applicationNovelty"],
    ["dataset_or_scenario_novelty", "novelty.dimension.datasetNovelty"],
    ["evaluation_novelty", "novelty.dimension.evaluationNovelty"],
    ["combination_novelty", "novelty.dimension.combinationNovelty"],
  ]
    .map(([key, label]) => [key, label, dimensions?.[key]])
    .filter(([, , value]) => value && typeof value === "object");
  if (!rows.length) return null;
  return (
    <div className="novelty-section novelty-dimension-summary">
      <h3>{t("novelty.dimensionSummaryTitle")}</h3>
      <div className="novelty-dimension-summary-grid">
        {rows.map(([key, label, value]) => (
          <details className={`novelty-dimension-card novelty-risk--${value.risk || "unknown"}`} key={key}>
            <summary>
              <strong>{t(label)}</strong>
              <span>{t(`novelty.risk.${value.risk || "unknown"}`)}</span>
            </summary>
            <p>{value.assessment || ""}</p>
          </details>
        ))}
      </div>
    </div>
  );
}

function PriorityPriorWork({ rows, comparisons, t }) {
  const priorRows = rows.slice(0, 3).map((row, index) => ({
    title: row.title,
    source: row.source,
    year: row.year,
    risk: row.risk || "unknown",
    meta: [row.year, row.verification_status].filter(Boolean).join(" / "),
    overlap: row.key_overlap,
    delta: row.key_delta,
    key: `${row.reference_index}-${row.title}-${index}`,
  }));
  const fallbackRows = comparisons.slice(0, 3).map((comparison, index) => ({
    title: comparison.title,
    source: comparison.source,
    risk: riskFromOverlap(comparison.overlap_level),
    meta: [comparison.authors, comparison.year, comparison.source_label].filter(Boolean).join(" / "),
    overlap: firstListItem(comparison.overlap_points),
    delta: firstListItem(comparison.difference_points),
    key: `${comparison.reference_index}-${comparison.title}-${index}`,
  }));
  const items = priorRows.length ? priorRows : fallbackRows;
  if (!items.length) return null;
  return (
    <div className="novelty-section novelty-priority-work">
      <h3>{t("novelty.priorityTitle")}</h3>
      <div className="novelty-priority-grid">
        {items.map((item) => (
          <article className="novelty-priority-card" key={item.key}>
            <span className={`novelty-risk novelty-risk--${item.risk}`}>{t(`novelty.risk.${item.risk}`)}</span>
            <h4>
              <ReferenceLink title={item.title || t("candidate.untitled")} source={item.source || ""} t={t} />
            </h4>
            {item.meta ? <p className="candidate-byline">{item.meta}</p> : null}
            {item.overlap ? <p><strong>{t("novelty.overlapPoints")}</strong>{item.overlap}</p> : null}
            {item.delta ? <p><strong>{t("novelty.differencePoints")}</strong>{item.delta}</p> : null}
          </article>
        ))}
      </div>
    </div>
  );
}

function riskFromOverlap(overlapLevel) {
  const level = normalizedOverlapLevel(overlapLevel);
  if (level === "high_overlap") return "high";
  if (level === "partial_overlap") return "moderate";
  if (level === "adjacent") return "low";
  return "unknown";
}

function ClosestPriorWork({ rows, t }) {
  if (!rows.length) return null;
  return (
    <div className="novelty-section novelty-closest-prior">
      <h3>{t("novelty.closestPriorTitle")}</h3>
      <div className="novelty-query-table novelty-closest-table" role="table" aria-label={t("novelty.closestPriorTitle")}>
        <div className="novelty-query-row novelty-query-row--head" role="row">
          <span>{t("novelty.queryId")}</span>
          <span>{t("novelty.closestWork")}</span>
          <span>{t("novelty.overlapPoints")}</span>
          <span>{t("novelty.differencePoints")}</span>
          <span>{t("novelty.verificationStatus")}</span>
        </div>
        {rows.map((row, index) => (
          <div className="novelty-query-row" role="row" key={`${row.reference_index}-${row.title}-${index}`}>
            <span>{Number(row.reference_index ?? index) + 1}</span>
            <span title={row.source || ""}>{[row.title, row.year].filter(Boolean).join(" / ") || "-"}</span>
            <span>{row.key_overlap || "-"}</span>
            <span>{row.key_delta || "-"}</span>
            <span>{row.verification_status || "partial"} / {row.risk || "unknown"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DimensionList({ dimensions, t }) {
  const rows = [
    ["target_problem", "novelty.dimension.targetProblem"],
    ["data_or_population", "novelty.dimension.data"],
    ["method", "novelty.dimension.method"],
    ["application_context", "novelty.dimension.context"],
    ["evaluation", "novelty.dimension.evaluation"],
  ].filter(([key]) => dimensions?.[key]);
  if (!rows.length) return null;
  return (
    <div className="novelty-dimension-list">
      <strong>{t("novelty.dimensionTitle")}</strong>
      <div>
        {rows.map(([key, label]) => (
          <span key={key}>{t(label)}: {t(`novelty.dimension.${dimensions[key]}`)}</span>
        ))}
      </div>
    </div>
  );
}

function DetailList({ title, items }) {
  if (!items.length) return null;
  return (
    <div>
      <strong>{title}</strong>
      <ul>
        {items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
      </ul>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="novelty-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
