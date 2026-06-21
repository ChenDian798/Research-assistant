import documentSearchIcon from "../assets/web-icons/research-assistant/document-search.svg";
import doiLinkIcon from "../assets/web-icons/research-assistant/doi-link.svg";
import emptyPaperIcon from "../assets/web-icons/research-assistant/empty-state-paper.svg";
import uploadCloudIcon from "../assets/web-icons/research-assistant/upload-cloud.svg";
import analysisIcon from "../assets/web-icons/search-flow/analysis.svg";
import searchIcon from "../assets/web-icons/search-flow/search.svg";
import startIcon from "../assets/web-icons/search-flow/start.svg";
import uploadIcon from "../assets/web-icons/search-flow/upload.svg";

const iconToneClass = {
  blue: "direct-icon--blue",
  mint: "direct-icon--mint",
  coral: "direct-icon--coral",
  neutral: "direct-icon--neutral",
};

const iconSrc = {
  add: documentSearchIcon,
  analyze: analysisIcon,
  file: uploadIcon,
  link: doiLinkIcon,
  search: searchIcon,
  start: startIcon,
  upload: uploadCloudIcon,
};

const watermarkSrc = {
  file: emptyPaperIcon,
  link: doiLinkIcon,
};

function iconClassName(type, tone = "blue", size = "md", extra = "") {
  return [
    "direct-icon",
    `direct-icon--${type}`,
    `direct-icon--${size}`,
    iconToneClass[tone] || iconToneClass.blue,
    extra,
  ].filter(Boolean).join(" ");
}

export function DirectIcon({ type, tone = "blue", size = "md", className = "" }) {
  return (
    <span className={iconClassName(type, tone, size, className)} aria-hidden="true">
      <img className="direct-icon__asset" src={iconSrc[type] || analysisIcon} alt="" />
    </span>
  );
}

export function SectionHeading({ icon, tone, title, children }) {
  return (
    <div className="direct-section-heading">
      <DirectIcon type={icon} tone={tone} size="lg" />
      <div>
        <h2>{title}</h2>
        <p>{children}</p>
      </div>
    </div>
  );
}

export function UploadDropIllustration() {
  return (
    <span className="upload-drop-illustration" aria-hidden="true">
      <img className="upload-drop-illustration__image" src={uploadCloudIcon} alt="" />
      <span className="upload-drop-illustration__spark upload-drop-illustration__spark--left" />
      <span className="upload-drop-illustration__spark upload-drop-illustration__spark--right" />
    </span>
  );
}

export function SummaryWatermark({ type }) {
  return (
    <span className={`direct-summary-watermark direct-summary-watermark--${type}`} aria-hidden="true">
      <img src={watermarkSrc[type] || analysisIcon} alt="" />
    </span>
  );
}

export function SummaryCard({ type, tone, title, children, isEmpty }) {
  return (
    <div className={`direct-summary-card direct-summary-card--${type}`}>
      <DirectIcon type={type} tone={tone} size="sm" />
      <div className="direct-summary-copy">
        <strong>{title}</strong>
        <div className={`direct-summary-content ${isEmpty ? "empty-state" : ""}`}>{children}</div>
      </div>
      <SummaryWatermark type={type} />
    </div>
  );
}

export function ButtonGlyph({ type }) {
  if (type === "analyze") {
    return <img className="direct-button-glyph direct-button-glyph--asset" src={startIcon} alt="" aria-hidden="true" />;
  }

  return null;
}
