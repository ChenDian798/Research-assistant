import FileUploadCard from "../components/FileUploadCard.jsx";
import LinkInputCard from "../components/LinkInputCard.jsx";
import {
  ButtonGlyph,
  SectionHeading,
  SummaryCard,
} from "../components/DirectAnalysisWidgets.jsx";

export default function DirectAnalysisView({
  status,
  statusError,
  doiInput,
  isRunning,
  selectedPdfFiles,
  pdfSummary,
  linkSummary,
  onDoiInputChange,
  onFilesSelected,
  onRemoveFile,
  onClearFiles,
  onAnalyze,
  onStartNewTask,
  showStartNewTask = false,
  hasResult,
  t,
}) {
  const hasLinks = Boolean(doiInput.trim());
  const hasFiles = selectedPdfFiles.length > 0;
  const hasDirectInput = hasLinks || hasFiles;
  const hasBackendStatus = Boolean(status);
  const inputStatus = hasFiles && hasLinks ? t("status.fileAndLinkReady") : (hasFiles ? t("status.fileReady") : t("status.linkReady"));
  const currentStatus = statusError
    ? status
    : (isRunning
      ? t("status.analyzing")
      : (hasDirectInput ? inputStatus : (hasBackendStatus ? status : t("status.waitingInput"))));
  const statusTone = statusError ? "error" : (isRunning ? "running" : (hasDirectInput ? "ready" : "idle"));

  return (
    <section id="directView" className="app-view direct-view is-active">
      <div className="direct-hero split-entry-hero split-entry-hero--direct">
        <div className="direct-hero-copy split-entry-hero__copy">
          <p className="eyebrow direct-eyebrow">Direct Analysis</p>
          <h1>{t("direct.title")}</h1>
          <p className="view-lead">{t("direct.lead")}</p>
        </div>
        <div className="split-entry-hero__stage">
          <div className="direct-header-actions">
            <span className={`analysis-status direct-status-pill direct-status-pill--${statusTone}`} role="status" aria-live="polite">
              {currentStatus}
            </span>
            {showStartNewTask ? (
              <button className="primary-button direct-new-task-button" type="button" onClick={onStartNewTask}>
                {t("direct.startNewTask")}
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

      <div className="direct-layout">
        <section className="panel direct-add-panel">
          <SectionHeading icon="add" tone="blue" title={t("direct.addTitle")}>
            {t("direct.addBody")}
          </SectionHeading>
          <div className="panel-body direct-input-grid">
            <FileUploadCard disabled={isRunning} onFilesSelected={onFilesSelected} t={t} />
            <LinkInputCard value={doiInput} disabled={isRunning} onChange={onDoiInputChange} t={t} />
          </div>
        </section>

        <aside className="panel direct-summary-panel side-panel">
          <SectionHeading icon="search" tone="coral" title={t("direct.pendingTitle")}>
            {t("direct.pendingBody")}
          </SectionHeading>
          <div className="panel-body">
            <div className="input-summary-stack">
              <SummaryCard type="file" tone="blue" title={t("direct.file")} isEmpty={!selectedPdfFiles.length}>
                {selectedPdfFiles.length ? (
                  <div className="direct-file-summary">
                    <span className="direct-file-summary-text">{pdfSummary}</span>
                    <ul className="direct-file-list" aria-label={t("direct.addedFilesAria")}>
                      {selectedPdfFiles.map((file) => (
                        <li className="direct-file-item" key={`${file.name}:${file.size}:${file.lastModified}`}>
                          <span className="direct-file-name" title={file.name}>{file.name}</span>
                          <button
                            className="direct-file-remove"
                            type="button"
                            disabled={isRunning}
                            aria-label={t("direct.removeAria", { name: file.name })}
                            onClick={() => onRemoveFile(file)}
                          >
                            {t("direct.remove")}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : t("summary.noFiles")}
              </SummaryCard>
              <SummaryCard type="link" tone="mint" title={t("direct.linkId")} isEmpty={!hasLinks}>
                {hasLinks ? linkSummary : t("summary.noLinks")}
              </SummaryCard>
            </div>
            <div className="button-stack">
              <button id="doiAnalyzeButton" className="primary-button direct-analyze-button" type="button" disabled={isRunning} onClick={onAnalyze}>
                <ButtonGlyph type="analyze" />
                {isRunning ? t("status.analyzing") : (hasResult ? t("direct.reanalyze") : t("direct.startAnalysis"))}
              </button>
              <button id="clearPdfButton" className="ghost-button direct-clear-button full" type="button" disabled={isRunning || !selectedPdfFiles.length} onClick={onClearFiles}>
                <ButtonGlyph type="clear" />
                {t("direct.clearFiles")}
              </button>
            </div>
            <p className="direct-helper-note">{t("direct.helper")}</p>
          </div>
        </aside>
      </div>
    </section>
  );
}
