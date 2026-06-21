import LoadingState from "./LoadingState.jsx";
import ReferenceLink from "./ReferenceLink.jsx";
import { innovationCellValue, publicCellText, toStringList } from "../lib/formatters.js";

export default function AnalysisTable({ rows, displayReferences, reviewNeededDocuments, mode, errorMessage, t }) {
  return (
    <div className="analysis-table-scroll">
      <table className="analysis-table doi-table">
        <thead>
          <tr>
            <th>{t("table.reference")}</th>
            <th>{t("table.contribution")}</th>
            <th>{t("table.methodEvidence")}</th>
            <th>{t("table.evidenceStrength")}</th>
            <th>{t("table.innovation")}</th>
            <th>{t("table.limitations")}</th>
            <th>{t("table.positioning")}</th>
            <th>{t("table.nextSteps")}</th>
          </tr>
        </thead>
        <tbody>
          {mode === "loading" ? (
            <tr className="analysis-loading-row">
              <td colSpan="8" className="analysis-loading-cell">
                <LoadingState
                  title={t("table.loadingTitle")}
                  message={displayReferences?.length ? t("table.loadingWithCount", { count: displayReferences.length }) : t("table.loadingDefault")}
                />
              </td>
            </tr>
          ) : mode === "interrupted" ? (
            <InterruptedRow message={errorMessage} t={t} />
          ) : mode === "error" ? (
            <ErrorRows references={displayReferences} message={errorMessage} t={t} />
          ) : rows.length || reviewNeededDocuments.length ? (
            <>
              {rows.map((row, index) => <AnalysisRow row={row} fallback={displayReferences[index] || {}} t={t} key={`${row.title || "row"}-${index}`} />)}
              {reviewNeededDocuments.map((document, index) => <ReviewNeededRow document={document} t={t} key={`${document.title || "review"}-${index}`} />)}
            </>
          ) : mode === "done" ? (
            <tr><td colSpan="8" className="empty-state">{t("table.doneEmpty")}</td></tr>
          ) : (
            <tr><td colSpan="8" className="empty-state">{t("table.idleEmpty")}</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function AnalysisRow({ row, fallback, t }) {
  return (
    <tr>
      <th><ReferenceLink title={row.title || fallback.title || t("table.untitled")} source={row.source || fallback.source || ""} uploadedFilename={row.uploaded_filename || fallback.uploaded_filename || ""} t={t} /></th>
      <td>{reviewCell(row.contribution || row.innovation, t("table.unclear"))}</td>
      <td>{reviewCell(row.methodology || row.method, t("table.unclear"))}</td>
      <td>{reviewCell(row.evidence_strength, t("table.unclear"))}</td>
      <td>{reviewCell(innovationCellValue(row), t("table.unclear"))}</td>
      <td>{reviewCell(row.limitations || row.weaknesses || row.limitation, t("table.authorUnclear"))}</td>
      <td>{reviewCell(row.literature_positioning, t("table.unclear"))}</td>
      <td>
        {reviewCell(row.actionable_suggestions || row.next_step || t("table.done"), t("table.unclear"))}
        {row.confidence ? <><br /><small>{row.confidence}</small></> : null}
      </td>
    </tr>
  );
}

function ReviewNeededRow({ document, t }) {
  return (
    <tr>
      <th><ReferenceLink title={document.title || t("table.untitled")} source={document.source || ""} uploadedFilename={document.uploaded_filename || ""} t={t} /></th>
      <td>{t("table.reviewNeeded")}</td>
      <td>{reviewCell(document.review_note || t("table.reviewNote"), t("table.unclear"))}</td>
      <td>{t("table.notAnalyzed")}</td>
      <td>{reviewCell(toStringList(document.review_reasons).join("；") || document.pdf_identity_status || "needs_review", t("table.unclear"))}</td>
      <td>{t("table.manualCheck")}</td>
      <td>{t("table.excluded")}</td>
      <td>{t("table.reupload")}</td>
    </tr>
  );
}

function InterruptedRow({ message, t }) {
  return (
    <tr className="analysis-interrupted-row">
      <td colSpan="8" className="analysis-interrupted-cell">
        <div className="analysis-interrupted-state" role="status">
          <span aria-hidden="true" />
          <div>
            <strong>{t("table.interruptedTitle")}</strong>
            <p>{message || t("table.interruptedBody")}</p>
          </div>
        </div>
      </td>
    </tr>
  );
}

function ErrorRows({ references = [], message, t }) {
  const items = references.length ? references : [{ title: t("table.untitled") }];
  return items.map((reference, index) => (
    <tr key={`error-${index}`}>
      <th><ReferenceLink title={reference.title || t("table.untitled")} source={reference.source || ""} uploadedFilename={reference.uploaded_filename || ""} t={t} /></th>
      <td>{t("table.analysisFailed")}</td>
      <td>{t("table.notGenerated")}</td>
      <td>{t("table.notGenerated")}</td>
      <td>{t("table.notGenerated")}</td>
      <td>{t("table.notGenerated")}</td>
      <td>{message}</td>
      <td>{t("table.failed")}</td>
    </tr>
  ));
}

function reviewCell(value, unclearLabel) {
  return publicCellText(value, unclearLabel);
}
