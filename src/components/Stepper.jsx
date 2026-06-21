import stepSearchIcon from "../assets/web-icons/search-flow/step-1-search.svg";
import stepSelectIcon from "../assets/web-icons/search-flow/step-2-select.svg";
import stepConfirmIcon from "../assets/web-icons/search-flow/step-3-confirm.svg";
import stepAnalyzeIcon from "../assets/web-icons/search-flow/step-4-analyze.svg";

const steps = [
  [1, "stepper.search", "stepper.searchSub", stepSearchIcon],
  [2, "stepper.select", "stepper.selectSub", stepSelectIcon],
  [3, "stepper.confirm", "stepper.confirmSub", stepConfirmIcon],
  [4, "stepper.analyze", "stepper.analyzeSub", stepAnalyzeIcon],
];

export default function Stepper({ activeStep, onStepChange, t }) {
  return (
    <nav className="stepper" aria-label={t("stepper.aria")}>
      {steps.map(([step, titleKey, subtitleKey, iconSrc]) => (
        <button
          className={`step ${activeStep === step ? "is-active" : ""} ${activeStep > step ? "is-done" : ""}`}
          type="button"
          onClick={() => onStepChange(step)}
          key={step}
        >
          <span className="step-number">
            <img className="step-icon" src={iconSrc} alt="" aria-hidden="true" />
            <span className="step-index">{step}</span>
          </span>
          <span><strong>{t(titleKey)}</strong><small>{t(subtitleKey)}</small></span>
        </button>
      ))}
    </nav>
  );
}
