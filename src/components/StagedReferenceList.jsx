import { referenceIdentifierText } from "../lib/formatters.js";

export default function StagedReferenceList({ references, onRemove, t }) {
  if (!references.length) {
    return (
      <div className="staged-reference-list empty-selection">
        <strong>{t("staged.emptyTitle")}</strong>
        <p>{t("staged.emptyBody")}</p>
      </div>
    );
  }
  return (
    <div className="staged-reference-list">
      <strong>{t("staged.count", { count: references.length })}</strong>
      <div className="selected-reference-list">
        {references.map((reference, index) => {
          const identifier = translatedIdentifier(referenceIdentifierText(reference), t);
          return (
            <article className="selected-reference-item" key={`${referenceIdentifierText(reference)}-${index}`}>
              <div>
                <span className="selected-reference-index">{index + 1}</span>
                <strong>{reference.title || t("candidate.untitled")}</strong>
                <small>{identifier} · {reference.verification_status || "partial"}</small>
              </div>
              <button className="ghost-button selected-reference-remove" type="button" onClick={() => onRemove(index)}>{t("direct.remove")}</button>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function translatedIdentifier(identifier, t) {
  return identifier === "无稳定 ID" ? t("reference.noStableId") : identifier;
}
