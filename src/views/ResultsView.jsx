import AnalysisTable from "../components/AnalysisTable.jsx";
import LiteratureSummary from "../components/LiteratureSummary.jsx";
import Stepper from "../components/Stepper.jsx";

export default function ResultsView({
  result,
  mode,
  errorMessage,
  exportFormat,
  exportStatusError,
  showSearchFlowStepper = false,
  searchStep = 4,
  onSearchStepChange,
  onExportFormatChange,
  onExport,
  onStartNewTask,
  showStartNewTask = false,
  t,
}) {
  const hasText = Boolean(result.rows.length || result.summary);
  const hasRows = Boolean(result.rows.length);
  const hasResultContent = Boolean(result.rows.length || result.summary || result.reviewNeededDocuments.length);
  const tableMode = mode === "loading" && hasResultContent ? "done" : mode;

  return (
    <section id="resultsView" className="app-view is-active">
      <div className="view-header">
        <div>
          <p className="eyebrow">Analysis Result</p>
          <h1>{t("results.title")}</h1>
          <p className="view-lead">{t("results.lead")}</p>
        </div>
        <div className="result-actions">
          <label className={`inline-export-select ${hasText ? "" : "is-hidden"}`}>
            <span>{t("results.format")}</span>
            <select value={exportFormat} disabled={!hasText} onChange={(event) => onExportFormatChange(event.target.value)}>
              <option value="md">MD</option>
              <option value="txt">TXT</option>
              <option value="pdf">PDF</option>
            </select>
          </label>
          <button className={`primary-button ${hasText ? "" : "is-hidden"}`} type="button" disabled={!hasText} onClick={onExport}>{t("results.export")}</button>
          {showStartNewTask ? (
            <button className="primary-button result-new-task-button" type="button" onClick={onStartNewTask}>
              {t("direct.startNewTask")}
            </button>
          ) : null}
        </div>
      </div>

      {showSearchFlowStepper ? (
        <Stepper activeStep={searchStep} onStepChange={onSearchStepChange} t={t} />
      ) : null}

      <section className={`panel ${exportStatusError ? "export-error" : ""}`}>
        <LiteratureSummary summary={result.summary} t={t} />
        <AnalysisTable
          rows={result.rows}
          displayReferences={result.displayReferences}
          reviewNeededDocuments={result.reviewNeededDocuments}
          mode={tableMode}
          errorMessage={errorMessage}
          t={t}
        />
      </section>
    </section>
  );
}
