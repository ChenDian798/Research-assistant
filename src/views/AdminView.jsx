import { useEffect, useMemo, useState } from "react";
import { createAdminUser, deleteAdminUser, fetchAdminUsers, updateAdminUser } from "../lib/api.js";

export default function AdminView({ currentUser, t }) {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [isError, setIsError] = useState(false);
  const [createForm, setCreateForm] = useState({ email: "", displayName: "", password: "", role: "user" });
  const [temporaryPassword, setTemporaryPassword] = useState("");

  const metrics = useMemo(() => {
    return {
      total: users.length,
      admins: users.filter((user) => user.role === "admin").length,
      disabled: users.filter((user) => user.status === "disabled").length,
    };
  }, [users]);

  async function loadUsers() {
    setIsLoading(true);
    setIsError(false);
    try {
      setUsers(await fetchAdminUsers());
      setStatus("");
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadUsers();
  }, []);

  async function patchUser(user, patch, successMessage) {
    setStatus(t("admin.saving"));
    setIsError(false);
    try {
      await updateAdminUser(user.id, patch);
      await loadUsers();
      setStatus(successMessage);
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    }
  }

  async function handleCreateUser(event) {
    event.preventDefault();
    setStatus(t("admin.creating"));
    setTemporaryPassword("");
    setIsError(false);
    try {
      const payload = await createAdminUser(createForm);
      await loadUsers();
      setTemporaryPassword(payload.temporary_password || "");
      setCreateForm({ email: "", displayName: "", password: "", role: "user" });
      setStatus(t("admin.created", { email: payload.user?.email || "" }));
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    }
  }

  async function removeUser(user) {
    const label = user.email || user.display_name || user.id;
    if (!window.confirm(t("admin.deleteConfirm", { email: label }))) return;
    setStatus(t("admin.deleting"));
    setIsError(false);
    try {
      await deleteAdminUser(user.id);
      await loadUsers();
      setStatus(t("admin.deleted"));
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    }
  }

  return (
    <section id="adminView" className="app-view admin-view is-active">
      <div className="view-header">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>{t("admin.title")}</h1>
          <p className="view-lead">{t("admin.lead")}</p>
        </div>
        <button className="ghost-button" type="button" onClick={loadUsers} disabled={isLoading}>
          {isLoading ? t("admin.refreshing") : t("admin.refresh")}
        </button>
      </div>

      <div className="admin-metrics">
        <Metric label={t("admin.totalUsers")} value={metrics.total} />
        <Metric label={t("admin.adminUsers")} value={metrics.admins} />
        <Metric label={t("admin.disabledUsers")} value={metrics.disabled} />
      </div>

      {status ? <p className={`analysis-status ${isError ? "error" : ""}`}>{status}</p> : null}
      {temporaryPassword ? (
        <div className="admin-created-password" role="status">
          <span>{t("admin.temporaryPassword")}</span>
          <strong>{temporaryPassword}</strong>
        </div>
      ) : null}

      <section className="panel admin-panel">
        <div className="panel-header">
          <div>
            <h2>{t("admin.createUser")}</h2>
            <p>{t("admin.createUserLead")}</p>
          </div>
        </div>
        <form className="panel-body admin-create-form" onSubmit={handleCreateUser}>
          <label className="auth-field">
            <span>{t("auth.email")}</span>
            <input value={createForm.email} onChange={(event) => setCreateForm((current) => ({ ...current, email: event.target.value }))} type="email" required />
          </label>
          <label className="auth-field">
            <span>{t("auth.displayName")}</span>
            <input value={createForm.displayName} onChange={(event) => setCreateForm((current) => ({ ...current, displayName: event.target.value }))} />
          </label>
          <label className="auth-field">
            <span>{t("admin.initialPassword")}</span>
            <input value={createForm.password} onChange={(event) => setCreateForm((current) => ({ ...current, password: event.target.value }))} type="text" minLength={8} placeholder={t("admin.passwordPlaceholder")} />
          </label>
          <label className="auth-field">
            <span>{t("admin.role")}</span>
            <select value={createForm.role} onChange={(event) => setCreateForm((current) => ({ ...current, role: event.target.value }))}>
              <option value="user">{t("admin.roleUser")}</option>
              <option value="admin">{t("admin.roleAdmin")}</option>
            </select>
          </label>
          <div className="form-actions admin-create-actions">
            <button className="primary-button" type="submit">{t("admin.createAction")}</button>
          </div>
        </form>
      </section>

      <section className="panel admin-panel">
        <div className="panel-header">
          <div>
            <h2>{t("admin.userList")}</h2>
            <p>{t("admin.userListLead")}</p>
          </div>
        </div>
        <div className="admin-table-scroll">
          <table className="analysis-table admin-table">
            <thead>
              <tr>
                <th>{t("admin.account")}</th>
                <th>{t("admin.role")}</th>
                <th>{t("admin.status")}</th>
                <th>{t("admin.activity")}</th>
                <th>{t("admin.createdAt")}</th>
                <th>{t("admin.actions")}</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => {
                const isSelf = user.id === currentUser?.id;
                const label = user.email || user.display_name || user.id;
                return (
                  <tr key={user.id}>
                    <th>
                      <strong>{label}</strong>
                      <span>{user.display_name && user.display_name !== user.email ? user.display_name : user.id}</span>
                    </th>
                    <td><span className={`history-status history-status--${user.role === "admin" ? "searched" : "done"}`}>{roleLabel(user.role, t)}</span></td>
                    <td><span className={`history-status history-status--${user.status === "active" ? "done" : "error"}`}>{statusLabel(user.status, t)}</span></td>
                    <td>{t("admin.activityValue", { history: user.history_count || 0, jobs: user.job_count || 0, files: user.file_count || 0 })}</td>
                    <td>{formatDateTime(user.created_at)}</td>
                    <td>
                      <div className="admin-row-actions">
                        <button className="ghost-button" type="button" onClick={() => patchUser(user, { status: user.status === "active" ? "disabled" : "active" }, t("admin.saved"))} disabled={isSelf && user.status === "active"}>
                          {user.status === "active" ? t("admin.disable") : t("admin.enable")}
                        </button>
                        <button className="ghost-button" type="button" onClick={() => patchUser(user, { role: user.role === "admin" ? "user" : "admin" }, t("admin.saved"))} disabled={isSelf}>
                          {user.role === "admin" ? t("admin.makeUser") : t("admin.makeAdmin")}
                        </button>
                        <button className="ghost-button danger" type="button" onClick={() => removeUser(user)} disabled={isSelf}>
                          {t("admin.delete")}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {!users.length && !isLoading ? (
                <tr>
                  <td colSpan="6">{t("admin.empty")}</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="history-meta-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function roleLabel(role, t) {
  return role === "admin" ? t("admin.roleAdmin") : t("admin.roleUser");
}

function statusLabel(status, t) {
  return status === "disabled" ? t("admin.statusDisabled") : t("admin.statusActive");
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}
