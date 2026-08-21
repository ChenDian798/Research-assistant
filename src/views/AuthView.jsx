import { useState } from "react";

export default function AuthView({ onLogin, onRegister, t }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [status, setStatus] = useState("");
  const [isError, setIsError] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setIsSubmitting(true);
    setStatus(mode === "login" ? t("auth.loggingIn") : t("auth.registering"));
    setIsError(false);
    try {
      if (mode === "login") {
        await onLogin(email, password);
      } else {
        await onRegister({ email, password, displayName });
      }
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="app-view auth-view is-active" aria-label={t("auth.title")}>
      <div className="auth-layout">
        <div className="auth-copy">
          <p className="eyebrow">{t("auth.eyebrow")}</p>
          <h1>{mode === "login" ? t("auth.loginTitle") : t("auth.registerTitle")}</h1>
          <p className="view-lead">{t("auth.lead")}</p>
        </div>
        <form className="panel auth-panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <div>
              <h2>{mode === "login" ? t("auth.loginPanel") : t("auth.registerPanel")}</h2>
              <p>{t("auth.panelLead")}</p>
            </div>
          </div>
          <div className="panel-body auth-form">
            {mode === "register" ? (
              <label className="auth-field">
                <span>{t("auth.displayName")}</span>
                <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete="name" />
              </label>
            ) : null}
            <label className="auth-field">
              <span>{t("auth.email")}</span>
              <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" autoComplete="email" required />
            </label>
            <label className="auth-field">
              <span>{t("auth.password")}</span>
              <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} minLength={8} required />
            </label>
            {status ? <p className={`analysis-status ${isError ? "error" : ""}`}>{status}</p> : null}
            <div className="form-actions">
              <button className="primary-button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? t("auth.submitting") : mode === "login" ? t("auth.loginAction") : t("auth.registerAction")}
              </button>
              <button
                className="ghost-button"
                type="button"
                onClick={() => {
                  setMode(mode === "login" ? "register" : "login");
                  setStatus("");
                  setIsError(false);
                }}
              >
                {mode === "login" ? t("auth.switchRegister") : t("auth.switchLogin")}
              </button>
            </div>
          </div>
        </form>
      </div>
    </section>
  );
}
