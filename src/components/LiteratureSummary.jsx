import { useRef, useState } from "react";
import { normalizeLiteratureSummary } from "../lib/formatters.js";

export default function LiteratureSummary({ summary, t }) {
  const normalized = normalizeLiteratureSummary(summary);
  const containerRef = useRef(null);
  const [height, setHeight] = useState(360);

  if (!normalized) {
    return <section id="doiSummary" className="literature-summary is-empty" aria-live="polite"></section>;
  }

  const groups = [
    [t("summary.commonStrengths"), normalized.common_strengths],
    [t("summary.commonWeaknesses"), normalized.common_weaknesses],
    [t("summary.methodPatterns"), normalized.methodological_patterns],
    [t("summary.evidenceGaps"), normalized.evidence_gaps],
    [t("summary.researchGaps"), normalized.research_gaps],
    [t("summary.readingOrder"), normalized.recommended_reading_order],
    [t("summary.references"), normalized.references],
    [t("summary.factRisks"), normalized.fact_risks],
    [t("summary.nextActions"), normalized.next_actions],
  ].filter(([, items]) => items.length);

  const resizeFromPointer = (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = containerRef.current?.getBoundingClientRect().height || height;
    document.body.classList.add("is-resizing-split");
    event.currentTarget.setPointerCapture(event.pointerId);

    const onMove = (moveEvent) => {
      const nextHeight = Math.max(140, Math.min(720, Math.round(startHeight + moveEvent.clientY - startY)));
      setHeight(nextHeight);
    };
    const onEnd = () => {
      document.body.classList.remove("is-resizing-split");
      event.currentTarget.removeEventListener("pointermove", onMove);
      event.currentTarget.removeEventListener("pointerup", onEnd);
      event.currentTarget.removeEventListener("pointercancel", onEnd);
    };

    event.currentTarget.addEventListener("pointermove", onMove);
    event.currentTarget.addEventListener("pointerup", onEnd);
    event.currentTarget.addEventListener("pointercancel", onEnd);
  };

  const resizeFromKeyboard = (event) => {
    if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let next = height;
    if (event.key === "ArrowUp") next -= 32;
    if (event.key === "ArrowDown") next += 32;
    if (event.key === "Home") next = 140;
    if (event.key === "End") next = 640;
    setHeight(Math.max(140, Math.min(720, Math.round(next))));
  };

  return (
    <>
      <section
        id="doiSummary"
        className="literature-summary is-resizable"
        aria-live="polite"
        ref={containerRef}
        style={{ "--summary-height": `${height}px` }}
      >
        <h3>{t("summary.title")}</h3>
        <p className="summary-lead">{normalized.overall_assessment || t("summary.generated")}</p>
        <div className="summary-grid">
          {groups.map(([title, items]) => <SummaryGroup title={title} items={items} key={title} />)}
          {normalized.confidence ? <SummaryGroup title={t("summary.confidence")} items={[normalized.confidence]} /> : null}
        </div>
      </section>
      <div
        className="split-resizer is-visible"
        role="separator"
        aria-orientation="horizontal"
        aria-label={t("summary.resizeAria")}
        tabIndex="0"
        onPointerDown={resizeFromPointer}
        onKeyDown={resizeFromKeyboard}
      ></div>
    </>
  );
}

function SummaryGroup({ title, items }) {
  return (
    <div className="summary-group">
      <strong>{title}</strong>
      <ul>
        {items.map((item, index) => <li key={`${title}-${index}`}>{item}</li>)}
      </ul>
    </div>
  );
}
