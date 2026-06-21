import {
  DirectIcon,
  UploadDropIllustration,
} from "./DirectAnalysisWidgets.jsx";

export default function FileUploadCard({ disabled, onFilesSelected, t }) {
  return (
    <section className="direct-input-card direct-input-card--upload">
      <div className="direct-input-title">
        <DirectIcon type="file" tone="blue" size="md" />
        <h3>{t("upload.title")}</h3>
      </div>
      <p>{t("upload.body")}</p>
      <label className="pdf-upload-box" htmlFor="pdfInput">
        <UploadDropIllustration />
        <strong>{t("upload.action")}</strong>
        <span>{t("upload.limit")}</span>
      </label>
      <input
        id="pdfInput"
        type="file"
        accept="application/pdf,.pdf,application/msword,.doc,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx"
        multiple
        disabled={disabled}
        onChange={(event) => {
          onFilesSelected(Array.from(event.target.files || []));
          event.target.value = "";
        }}
      />
    </section>
  );
}
