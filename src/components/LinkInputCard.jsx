import { DirectIcon } from "./DirectAnalysisWidgets.jsx";

export default function LinkInputCard({ value, disabled, onChange, t }) {
  return (
    <section className="direct-input-card direct-input-card--links">
      <div className="direct-input-title">
        <DirectIcon type="link" tone="mint" size="md" />
        <h3>{t("link.title")}</h3>
      </div>
      <p>{t("link.body")}</p>
      <textarea
        id="doiInput"
        rows="9"
        placeholder={"10.1038/s41586-024-xxxxx\nhttps://arxiv.org/abs/2501.00001\nhttps://pubmed.ncbi.nlm.nih.gov/12345678/"}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    </section>
  );
}
