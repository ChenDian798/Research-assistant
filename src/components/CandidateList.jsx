import ReferenceLink from "./ReferenceLink.jsx";
import { referenceIdentifierText, toStringList } from "../lib/formatters.js";

export default function CandidateList({ candidates, selectedIds, meta, onToggle, t }) {
  const qualified = candidates.filter((reference) => reference.candidate_group !== "needs_review");
  const needsReview = candidates.filter((reference) => reference.candidate_group === "needs_review");

  if (!candidates.length) {
    return (
      <>
        {meta ? <p id="searchCandidateMeta" className="candidate-meta">{meta}</p> : null}
        <div className="candidate-list">
          <div className="empty-state candidate-empty">{t("candidate.empty")}</div>
        </div>
      </>
    );
  }

  return (
    <>
      {meta ? <p id="searchCandidateMeta" className="candidate-meta">{meta}</p> : null}
      <div className="candidate-list">
        <CandidateGroup title={t("candidate.qualified")} references={qualified} selectedIds={selectedIds} onToggle={onToggle} t={t} />
        <CandidateGroup title={t("candidate.needsReview")} references={needsReview} selectedIds={selectedIds} onToggle={onToggle} t={t} />
      </div>
    </>
  );
}

function CandidateGroup({ title, references, selectedIds, onToggle, t }) {
  if (!references.length) return null;
  return (
    <section className="candidate-group">
      <h3>{title}</h3>
      {references.map((reference) => (
        <CandidateItem
          reference={reference}
          checked={selectedIds.has(reference.candidate_id)}
          onToggle={onToggle}
          t={t}
          key={reference.candidate_id}
        />
      ))}
    </section>
  );
}

function CandidateItem({ reference, checked, onToggle, t }) {
  const status = reference.candidate_group === "needs_review" ? "needs_review" : reference.screening_status;
  const risks = [...toStringList(reference.screening_risks), ...toStringList(reference.verification_risks)];
  const identifier = translatedIdentifier(referenceIdentifierText(reference), t);
  return (
    <article className={`candidate-item ${reference.candidate_group === "needs_review" ? "needs-review" : ""}`}>
      <label className="candidate-check">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onToggle(reference.candidate_id, event.target.checked)}
        />
        <span>
          <span className={`candidate-badge verification-${reference.verification_status || "partial"}`}>{reference.verification_status || "partial"}</span>{" "}
          <span className={`candidate-badge screening-${status || "qualified"}`}>{status || "qualified"}</span>
        </span>
      </label>
      <div className="candidate-main">
        <h3><ReferenceLink title={reference.title || t("candidate.untitled")} source={reference.source || ""} t={t} /></h3>
        <p className="candidate-byline">{[reference.authors, reference.year, reference.source_label].filter(Boolean).join(" · ") || t("candidate.incompleteMeta")}</p>
        <p className="candidate-idline">{identifier}</p>
        <p className="candidate-abstract">{reference.abstract || reference.relevance || t("candidate.noAbstract")}</p>
        {risks.length ? <p className="candidate-risks">{t("candidate.risks", { text: risks.join("；") })}</p> : null}
      </div>
    </article>
  );
}

function translatedIdentifier(identifier, t) {
  return identifier === "无稳定 ID" ? t("reference.noStableId") : identifier;
}
