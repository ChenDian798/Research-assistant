import ReferenceLink from "./ReferenceLink.jsx";
import { referenceIdentifierText, referenceStableKey, toStringList } from "../lib/formatters.js";

const CANDIDATE_BYLINE_MAX_CHARS = 260;
const CANDIDATE_ABSTRACT_MAX_CHARS = 900;

export default function CandidateList({ candidates, selectedIds, meta, onToggle, t, feedbackVotes = {}, feedbackPending = {}, onFeedback, selectionLocked = false }) {
  const qualified = candidates.filter((reference) => reference.candidate_group === "qualified");
  const needsReview = candidates.filter((reference) => reference.candidate_group === "needs_review");
  const rejected = candidates.filter((reference) => reference.candidate_group === "rejected");

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
        <CandidateGroup title={t("candidate.qualified")} references={qualified} selectedIds={selectedIds} onToggle={onToggle} t={t} feedbackVotes={feedbackVotes} feedbackPending={feedbackPending} onFeedback={onFeedback} selectionLocked={selectionLocked} />
        <CandidateGroup title={t("candidate.needsReview")} references={needsReview} selectedIds={selectedIds} onToggle={onToggle} t={t} feedbackVotes={feedbackVotes} feedbackPending={feedbackPending} onFeedback={onFeedback} selectionLocked={selectionLocked} />
        <CandidateGroup title={t("candidate.rejectedReferences")} references={rejected} selectedIds={selectedIds} onToggle={onToggle} t={t} feedbackVotes={feedbackVotes} feedbackPending={feedbackPending} onFeedback={onFeedback} selectable={false} />
      </div>
    </>
  );
}

function CandidateGroup({ title, references, selectedIds, onToggle, t, feedbackVotes, feedbackPending, onFeedback, selectable = true, selectionLocked = false }) {
  if (!references.length) return null;
  return (
    <section className="candidate-group">
      <h3>{title}</h3>
      {references.map((reference) => (
        <CandidateItem
          reference={reference}
          checked={selectable && selectedIds.has(reference.candidate_id)}
          selectionLocked={selectionLocked}
          onToggle={onToggle}
          t={t}
          feedbackVote={referenceFeedbackValue(feedbackVotes, reference)}
          feedbackPending={Boolean(referenceFeedbackValue(feedbackPending, reference))}
          onFeedback={onFeedback}
          selectable={selectable}
          key={reference.candidate_id}
        />
      ))}
    </section>
  );
}

function CandidateItem({ reference, checked, selectionLocked = false, onToggle, t, feedbackVote = "", feedbackPending = false, onFeedback, selectable = true }) {
  const status = reference.candidate_group === "needs_review" ? "needs_review" : reference.screening_status || reference.candidate_group;
  const risks = visibleRiskItems(reference);
  const identifier = translatedIdentifier(referenceIdentifierText(reference), t);
  const itemClass = [
    "candidate-item",
    reference.candidate_group === "needs_review" ? "needs-review" : "",
    reference.candidate_group === "rejected" ? "rejected-reference" : "",
  ].filter(Boolean).join(" ");
  const byline = [reference.authors, reference.year, reference.source_label].filter(Boolean).join(" · ") || t("candidate.incompleteMeta");
  const abstract = reference.abstract || reference.relevance || t("candidate.noAbstract");
  return (
    <article className={itemClass}>
      <label className="candidate-check">
        <input
          type="checkbox"
          checked={checked}
          disabled={!selectable || selectionLocked}
          onChange={(event) => selectable && !selectionLocked && onToggle(reference.candidate_id, event.target.checked)}
        />
        <span>
          <span className={`candidate-badge verification-${reference.verification_status || "partial"}`}>{reference.verification_status || "partial"}</span>{" "}
          <span className={`candidate-badge screening-${status || "qualified"}`}>{status || "qualified"}</span>
        </span>
      </label>
      <div className="candidate-main">
        <h3><ReferenceLink title={reference.title || t("candidate.untitled")} source={reference.source || ""} t={t} /></h3>
        <p className="candidate-byline">{truncateDisplayText(byline, CANDIDATE_BYLINE_MAX_CHARS)}</p>
        <p className="candidate-idline">{identifier}</p>
        <p className="candidate-abstract">{truncateDisplayText(abstract, CANDIDATE_ABSTRACT_MAX_CHARS)}</p>
        {risks.length ? <p className="candidate-risks">{t("candidate.risks", { text: risks.join("；") })}</p> : null}
      </div>
      {onFeedback ? (
        <div className="candidate-feedback" aria-label={t("feedback.label")}>
          <span>{t("feedback.prompt")}</span>
          <div className="candidate-feedback-actions">
            <button
              type="button"
              className={feedbackVote === "yes" ? "is-selected" : ""}
              disabled={feedbackPending}
              onClick={() => onFeedback(reference, "yes")}
            >
              {t("feedback.yes")}
            </button>
            <button
              type="button"
              className={feedbackVote === "no" ? "is-selected" : ""}
              disabled={feedbackPending}
              onClick={() => onFeedback(reference, "no")}
            >
              {t("feedback.no")}
            </button>
          </div>
        </div>
      ) : null}
    </article>
  );
}

function truncateDisplayText(value, maxChars) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= maxChars) return text;
  return `${text.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`;
}

function translatedIdentifier(identifier, t) {
  return identifier === "无稳定 ID" ? t("reference.noStableId") : identifier;
}

function referenceFeedbackValue(feedbackMap, reference) {
  const keys = Array.from(new Set([
    reference?.candidate_id,
    referenceStableKey(reference || {}),
  ].map((value) => String(value || "").trim()).filter(Boolean)));
  for (const key of keys) {
    if (feedbackMap[key]) return feedbackMap[key];
  }
  return "";
}

function visibleRiskItems(reference) {
  return Array.from(new Set([
    ...toStringList(reference.screening_reasons).filter(isActionableScreeningReason),
    ...toStringList(reference.screening_risks),
    ...toStringList(reference.topic_relevance_risks),
    ...toStringList(reference.verification_risks),
  ].filter(Boolean)));
}

function isActionableScreeningReason(reason) {
  return ![
    "has_abstract",
    "has_arxiv_id",
    "has_authors",
    "has_doi",
    "has_pmid",
    "has_stable_url",
    "has_year",
  ].includes(String(reason || "").trim());
}
