import researchAssistantMark from "../assets/research-assistant-mark.png";
import { languageOptions } from "../lib/i18n.js";

const navItems = [
  ["home", "nav.home"],
  ["direct", "nav.direct"],
  ["standaloneSearch", "nav.standaloneSearch"],
  ["novelty", "nav.novelty"],
  ["search", "nav.search"],
  ["history", "nav.history"],
];

export default function Topbar({ currentView, language, onLanguageChange, onNavigate, t }) {
  return (
    <header className="app-topbar" aria-label={t("nav.aria")}>
      <div className="brand">
        <img className="brand-logo-mark" src={researchAssistantMark} alt="" aria-hidden="true" />
        <span className="brand-wordmark" aria-label="Research Assistant">
          <span className="brand-wordmark-primary">Research</span>
          <span className="brand-wordmark-secondary">Assistant</span>
        </span>
      </div>
      <div className="topbar-actions">
        {navItems.map(([view, labelKey]) => (
          <button
            className={`ghost-button ${currentView === view ? "is-active" : ""}`}
            type="button"
            aria-current={currentView === view ? "page" : "false"}
            onClick={() => onNavigate(view)}
            key={view}
          >
            {t(labelKey)}
          </button>
        ))}
      </div>
      <div className="topbar-language" aria-label={t("nav.language")}>
        {languageOptions.map((option) => (
          <button
            className={`language-toggle-button ${language === option.value ? "is-active" : ""}`}
            type="button"
            aria-pressed={language === option.value}
            onClick={() => onLanguageChange(option.value)}
            key={option.value}
          >
            {option.label}
          </button>
        ))}
      </div>
    </header>
  );
}
