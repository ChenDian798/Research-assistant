import { useEffect, useMemo, useState } from "react";
import { createAdminUser, deleteAdminUser, fetchAdminAuditLogs, fetchAdminHistory, fetchAdminHistoryDetail, fetchAdminMetrics, fetchAdminUsers, updateAdminUser } from "../lib/api.js";

export default function AdminView({ currentUser, t }) {
  const [users, setUsers] = useState([]);
  const [metricsData, setMetricsData] = useState(null);
  const [metricsDays, setMetricsDays] = useState(7);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [auditLogs, setAuditLogs] = useState([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditActorUserId, setAuditActorUserId] = useState("");
  const [historyEntries, setHistoryEntries] = useState([]);
  const [historyFilters, setHistoryFilters] = useState({ ownerUserId: "", ownerKeyword: "", kind: "", status: "", limit: 100 });
  const [historyDetail, setHistoryDetail] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [isError, setIsError] = useState(false);
  const [createForm, setCreateForm] = useState({ email: "", displayName: "", password: "", role: "user" });
  const [temporaryPassword, setTemporaryPassword] = useState("");

  const accountMetrics = useMemo(() => {
    return {
      total: users.length,
      admins: users.filter((user) => user.role === "admin").length,
      disabled: users.filter((user) => user.status === "disabled").length,
    };
  }, [users]);

  const overviewMetrics = metricsData || emptyMetrics(metricsDays);
  const historyKinds = useMemo(() => uniqueValues(historyEntries.map((entry) => entry.kind)), [historyEntries]);
  const historyStatuses = useMemo(() => uniqueValues(historyEntries.map((entry) => entry.status)), [historyEntries]);

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

  async function loadMetrics(days = metricsDays) {
    setMetricsLoading(true);
    setIsError(false);
    try {
      setMetricsData(await fetchAdminMetrics(days));
      setStatus("");
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      setMetricsLoading(false);
    }
  }

  async function loadAuditLogs(actorUserId = auditActorUserId) {
    setAuditLoading(true);
    setIsError(false);
    try {
      setAuditLogs(await fetchAdminAuditLogs(30, actorUserId));
      setStatus("");
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      setAuditLoading(false);
    }
  }

  async function loadHistory(nextFilters = historyFilters) {
    setHistoryLoading(true);
    setIsError(false);
    try {
      const entries = await fetchAdminHistory(nextFilters);
      setHistoryEntries(entries);
      setHistoryDetail((current) => (current && entries.some((entry) => entry.id === current.id) ? current : null));
      setStatus("");
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      setHistoryLoading(false);
    }
  }

  useEffect(() => {
    loadUsers();
    loadMetrics(7);
    loadAuditLogs();
    loadHistory();
  }, []);

  function changeMetricsDays(days) {
    setMetricsDays(days);
    loadMetrics(days);
  }

  function refreshAdminData() {
    loadUsers();
    loadMetrics();
    loadAuditLogs();
    loadHistory();
  }

  function updateHistoryFilters(patch, loadImmediately = false) {
    const nextFilters = { ...historyFilters, ...patch };
    setHistoryFilters(nextFilters);
    if (loadImmediately) loadHistory(nextFilters);
  }

  function selectUserHistory(userId) {
    updateHistoryFilters({ ownerUserId: userId, ownerKeyword: "" }, true);
  }

  function changeAuditActor(userId) {
    setAuditActorUserId(userId);
    loadAuditLogs(userId);
  }

  async function openHistoryDetail(entry) {
    setHistoryLoading(true);
    setIsError(false);
    try {
      setHistoryDetail(await fetchAdminHistoryDetail(entry.id));
      setStatus("");
    } catch (error) {
      setStatus(error.message);
      setIsError(true);
    } finally {
      setHistoryLoading(false);
    }
  }

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
        <button className="ghost-button" type="button" onClick={refreshAdminData} disabled={isLoading || metricsLoading || historyLoading || auditLoading}>
          {isLoading || metricsLoading || historyLoading || auditLoading ? t("admin.refreshing") : t("admin.refresh")}
        </button>
      </div>

      <div className="admin-metrics">
        <Metric label={t("admin.totalUsers")} value={accountMetrics.total} />
        <Metric label={t("admin.adminUsers")} value={accountMetrics.admins} />
        <Metric label={t("admin.disabledUsers")} value={accountMetrics.disabled} />
      </div>

      {status ? <p className={`analysis-status ${isError ? "error" : ""}`}>{status}</p> : null}
      {temporaryPassword ? (
        <div className="admin-created-password" role="status">
          <span>{t("admin.temporaryPassword")}</span>
          <strong>{temporaryPassword}</strong>
        </div>
      ) : null}

      <section className="panel admin-panel admin-overview-panel">
        <div className="panel-header">
          <div>
            <h2>{t("admin.metricsTitle")}</h2>
            <p>{t("admin.metricsLead", { days: overviewMetrics.range_days || metricsDays })}</p>
          </div>
          <div className="admin-range-tabs" role="group" aria-label={t("admin.metricsRange")}>
            {[1, 7, 30].map((days) => (
              <button className={`ghost-button ${metricsDays === days ? "is-active" : ""}`} type="button" key={days} onClick={() => changeMetricsDays(days)} disabled={metricsLoading}>
                {t("admin.days", { days })}
              </button>
            ))}
          </div>
        </div>
        <div className="admin-overview-grid">
          <Metric label={t("admin.metricsUsers")} value={overviewMetrics.users.total || 0} />
          <Metric label={t("admin.metricsHistory")} value={overviewMetrics.usage.history_total || 0} />
          <Metric label={t("admin.metricsDoneJobs")} value={overviewMetrics.jobs.done || 0} />
          <Metric label={t("admin.metricsErrorJobs")} value={overviewMetrics.jobs.error || 0} />
          <Metric label={t("admin.metricsRunningJobs")} value={overviewMetrics.jobs.running || 0} />
          <Metric label={t("admin.metricsFiles")} value={overviewMetrics.storage.file_count || 0} />
        </div>
        <div className="admin-metrics-columns">
          <div className="admin-table-scroll">
            <table className="analysis-table admin-table admin-usage-table">
              <thead>
                <tr>
                  <th>{t("admin.account")}</th>
                  <th>{t("admin.metricsHistory")}</th>
                  <th>{t("admin.metricsDoneJobs")}</th>
                  <th>{t("admin.metricsErrorJobs")}</th>
                  <th>{t("admin.metricsFiles")}</th>
                  <th>{t("admin.lastActivity")}</th>
                  <th>{t("admin.actions")}</th>
                </tr>
              </thead>
              <tbody>
                {(overviewMetrics.users_usage || []).map((item) => (
                  <tr key={item.owner_user_id}>
                    <th>
                      <strong>{item.email || item.display_name || item.owner_user_id}</strong>
                      <span>{item.owner_user_id}</span>
                    </th>
                    <td>{item.history_total || 0}</td>
                    <td>{item.job_done || 0}</td>
                    <td>{item.job_error || 0}</td>
                    <td>{item.file_count || 0}</td>
                    <td>{formatDateTime(item.last_activity_at)}</td>
                    <td>
                      <button className="ghost-button" type="button" onClick={() => selectUserHistory(item.owner_user_id)}>
                        {t("admin.viewHistory")}
                      </button>
                    </td>
                  </tr>
                ))}
                {!(overviewMetrics.users_usage || []).length ? (
                  <tr>
                    <td colSpan="7">{t("admin.metricsUserEmpty")}</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
          <div className="admin-recent-errors">
            <h3>{t("admin.recentErrors")}</h3>
            <div className="admin-error-list">
              {(overviewMetrics.recent_errors || []).map((item) => (
                <button className="admin-error-item" type="button" key={item.history_id} onClick={() => openHistoryDetail({ id: item.history_id })}>
                  <strong>{item.user || item.owner_user_id}</strong>
                  <span>{item.kind || "-"} · {formatDateTime(item.updated_at)}</span>
                  <small>{item.error || item.title || item.history_id}</small>
                </button>
              ))}
              {!(overviewMetrics.recent_errors || []).length ? <p>{t("admin.recentErrorsEmpty")}</p> : null}
            </div>
          </div>
        </div>
      </section>

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
                        <button className="ghost-button" type="button" onClick={() => selectUserHistory(user.id)}>
                          {t("admin.viewHistory")}
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

      <section className="panel admin-panel admin-history-panel">
        <div className="panel-header">
          <div>
            <h2>{t("admin.historyTitle")}</h2>
            <p>{t("admin.historyLead")}</p>
          </div>
          <button className="ghost-button" type="button" onClick={() => loadHistory()} disabled={historyLoading}>
            {historyLoading ? t("admin.refreshing") : t("admin.refresh")}
          </button>
        </div>
        <div className="admin-history-filters">
          <label className="auth-field">
            <span>{t("admin.historyUser")}</span>
            <select value={historyFilters.ownerUserId} onChange={(event) => updateHistoryFilters({ ownerUserId: event.target.value }, true)}>
              <option value="">{t("admin.allUsers")}</option>
              {users.map((user) => (
                <option key={user.id} value={user.id}>{user.email || user.display_name || user.id}</option>
              ))}
            </select>
          </label>
          <label className="auth-field">
            <span>{t("admin.historyKeyword")}</span>
            <input value={historyFilters.ownerKeyword} onChange={(event) => updateHistoryFilters({ ownerKeyword: event.target.value })} onBlur={() => loadHistory()} placeholder={t("admin.historyKeywordPlaceholder")} />
          </label>
          <label className="auth-field">
            <span>{t("admin.historyKind")}</span>
            <select value={historyFilters.kind} onChange={(event) => updateHistoryFilters({ kind: event.target.value }, true)}>
              <option value="">{t("admin.any")}</option>
              {historyKinds.map((kind) => <option key={kind} value={kind}>{kind}</option>)}
            </select>
          </label>
          <label className="auth-field">
            <span>{t("admin.historyStatus")}</span>
            <select value={historyFilters.status} onChange={(event) => updateHistoryFilters({ status: event.target.value }, true)}>
              <option value="">{t("admin.any")}</option>
              {historyStatuses.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label className="auth-field">
            <span>{t("admin.historyLimit")}</span>
            <select value={historyFilters.limit} onChange={(event) => updateHistoryFilters({ limit: Number(event.target.value) }, true)}>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={300}>300</option>
            </select>
          </label>
        </div>
        <div className="admin-table-scroll">
          <table className="analysis-table admin-table admin-history-table">
            <thead>
              <tr>
                <th>{t("admin.historyOwner")}</th>
                <th>{t("admin.historyKind")}</th>
                <th>{t("admin.historyItemTitle")}</th>
                <th>{t("admin.historyStatus")}</th>
                <th>{t("admin.historyCounts")}</th>
                <th>{t("admin.updatedAt")}</th>
                <th>{t("admin.actions")}</th>
              </tr>
            </thead>
            <tbody>
              {historyEntries.map((entry) => (
                <tr key={entry.id}>
                  <th>
                    <strong>{entry.owner_email || entry.owner_display_name || entry.owner_user_id}</strong>
                    <span>{entry.owner_user_id}</span>
                  </th>
                  <td>{entry.kind || "-"}</td>
                  <td>
                    <strong>{historyTitle(entry)}</strong>
                    <span>{historyRequestLine(entry.request)}</span>
                  </td>
                  <td><span className={`history-status history-status--${entry.status || "done"}`}>{entry.status || "-"}</span></td>
                  <td>{formatCounts(entry.counts)}</td>
                  <td>{formatDateTime(entry.updated_at)}</td>
                  <td>
                    <button className="ghost-button" type="button" onClick={() => openHistoryDetail(entry)}>
                      {t("admin.historyDetail")}
                    </button>
                  </td>
                </tr>
              ))}
              {!historyEntries.length && !historyLoading ? (
                <tr>
                  <td colSpan="7">{t("admin.historyEmpty")}</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        {historyDetail ? (
          <div className="admin-history-detail">
            <div className="admin-history-detail-header">
              <p className="eyebrow">{historyDetail.owner_email || historyDetail.owner_user_id}</p>
              <h3>{historyDetail.title || historyDetail.request?.query || historyDetail.request?.topic || historyDetail.id}</h3>
              <p>{historyDetail.owner_user_id} · {historyDetail.kind || "-"} · {historyDetail.status || "-"}</p>
            </div>
            <AdminHistoryInsight detail={historyDetail} t={t} />
          </div>
        ) : null}
      </section>

      <section className="panel admin-panel admin-audit-panel">
        <div className="panel-header">
          <div>
            <h2>{t("admin.auditTitle")}</h2>
            <p>{t("admin.auditLead")}</p>
          </div>
          <button className="ghost-button" type="button" onClick={loadAuditLogs} disabled={auditLoading}>
            {auditLoading ? t("admin.refreshing") : t("admin.refresh")}
          </button>
        </div>
        <div className="admin-audit-filters">
          <label className="auth-field">
            <span>{t("admin.auditUser")}</span>
            <select value={auditActorUserId} onChange={(event) => changeAuditActor(event.target.value)}>
              <option value="">{t("admin.allUsers")}</option>
              {users.map((user) => (
                <option key={user.id} value={user.id}>{user.email || user.display_name || user.id}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="admin-table-scroll">
          <table className="analysis-table admin-table admin-audit-table">
            <thead>
              <tr>
                <th>{t("admin.auditTime")}</th>
                <th>{t("admin.auditActor")}</th>
                <th>{t("admin.auditAction")}</th>
                <th>{t("admin.auditSummary")}</th>
              </tr>
            </thead>
            <tbody>
              {auditLogs.map((item, index) => {
                const resource = auditResource(item, t);
                const detailItems = auditDetailItems(item.detail, t);
                return (
                  <tr key={`${item.created_at}-${item.action}-${item.resource_id}-${index}`}>
                    <td>{formatDateTime(item.created_at)}</td>
                    <th>
                      <strong>{item.actor_email || item.actor_display_name || item.actor_user_id || "-"}</strong>
                      <span>{shortAuditId(item.actor_user_id)}</span>
                    </th>
                    <td>
                      <strong>{auditActionLabel(item.action, t)}</strong>
                      <span className="admin-audit-raw">{item.action || "-"}</span>
                    </td>
                    <td className="admin-audit-summary">
                      <strong>
                        {resource.label}
                        <span title={item.resource_id || ""}>{resource.id}</span>
                      </strong>
                      {detailItems.length ? (
                        <div className="admin-audit-detail-list">
                          {detailItems.map(([label, value]) => (
                            <span key={label}><b>{label}:</b> {value}</span>
                          ))}
                        </div>
                      ) : <span>{t("admin.auditNoDetail")}</span>}
                    </td>
                  </tr>
                );
              })}
              {!auditLogs.length && !auditLoading ? (
                <tr>
                  <td colSpan="4">{t("admin.auditEmpty")}</td>
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

function AdminHistoryInsight({ detail, t }) {
  const summary = detail.admin_summary || buildHistoryInsight(detail);
  const request = detail.request || {};
  const counts = summary.counts || {};
  const timings = summary.timings || {};
  const analysisTiming = summary.analysis || {};
  const sourceResults = summary.source_results || {};
  const timingRows = [
    [t("admin.detailQueueTime"), formatDuration(summary.queue_seconds)],
    [t("admin.detailRunTime"), formatDuration(summary.run_seconds)],
    [t("admin.detailTotalTime"), formatDuration(summary.total_seconds ?? summary.history_elapsed_seconds)],
    [t("admin.detailPlanning"), formatDuration(timingValue(timings, "planning_seconds"))],
    [t("admin.detailRecall"), formatDuration(timingValue(timings, "recall_seconds"))],
    [t("admin.detailScreening"), formatDuration(timingValue(timings, "screening_seconds"))],
    [t("admin.detailVerification"), formatDuration(timingValue(timings, "verification_seconds"))],
    [t("admin.detailAnalysisQueueTime"), formatDuration(analysisTiming.queue_seconds)],
    [t("admin.detailAnalysisRunTime"), formatDuration(analysisTiming.run_seconds)],
    [t("admin.detailAnalysisTotalTime"), formatDuration(analysisTiming.total_seconds)],
  ].filter(([, value]) => value !== "-");
  const countRows = [
    [t("admin.detailCandidates"), firstCount(counts, "candidates", "candidate_count", "raw", "raw_count", "references")],
    [t("admin.detailQualified"), firstCount(counts, "qualified", "qualified_count")],
    [t("admin.detailNeedsReview"), firstCount(counts, "needs_review", "needs_review_count")],
    [t("admin.detailRejected"), firstCount(counts, "rejected", "rejected_count", "filtered", "filtered_count")],
    [t("admin.detailFiles"), firstCount(counts, "files", "file_count")],
    [t("admin.detailRows"), firstCount(counts, "rows", "row_count")],
  ].filter(([, value]) => value !== "-");
  const sourceRows = Object.entries(sourceResults).filter(([, value]) => value !== undefined && value !== null && value !== "");
  const errors = Array.isArray(summary.errors) ? summary.errors : [];
  const warnings = Array.isArray(summary.warnings) ? summary.warnings : [];

  return (
    <div className="admin-history-insight">
      <div className="admin-insight-grid">
        <InsightCard title={t("admin.detailTiming")} rows={timingRows} empty={t("admin.detailNoTiming")} />
        <InsightCard title={t("admin.detailCounts")} rows={countRows} empty={t("admin.detailNoCounts")} />
        <InsightCard title={t("admin.detailRequest")} rows={[
          [t("admin.detailQuery"), request.query || request.topic || request.innovation_text || "-"],
          [t("admin.detailSources"), request.sources || "-"],
          [t("admin.detailMode"), request.search_mode || "-"],
          [t("admin.detailYear"), request.year || "-"],
        ]} />
      </div>
      <div className="admin-insight-grid admin-insight-grid--secondary">
        <InsightCard title={t("admin.detailSourcesReturned")} rows={sourceRows.map(([key, value]) => [key, value])} empty={t("admin.detailNoSources")} />
        <InsightCard title={t("admin.detailIssues")} rows={[...errors.map((item) => [t("admin.detailError"), item]), ...warnings.map((item) => [t("admin.detailWarning"), item])]} empty={t("admin.detailNoIssues")} />
      </div>
      <details className="admin-raw-detail">
        <summary>{t("admin.detailRaw")}</summary>
        <pre>{JSON.stringify(detail, null, 2)}</pre>
      </details>
    </div>
  );
}

function InsightCard({ title, rows, empty = "-" }) {
  return (
    <section className="admin-insight-card">
      <h4>{title}</h4>
      {rows.length ? (
        <dl>
          {rows.map(([label, value], index) => (
            <div key={`${label}-${index}`}>
              <dt>{label}</dt>
              <dd>{String(value || "-")}</dd>
            </div>
          ))}
        </dl>
      ) : <p>{empty}</p>}
    </section>
  );
}

function emptyMetrics(days = 7) {
  return {
    range_days: days,
    users: { total: 0, active: 0, disabled: 0 },
    usage: { history_total: 0 },
    jobs: { queued: 0, running: 0, done: 0, error: 0 },
    quality: { candidate_total: 0, qualified_total: 0, needs_review_total: 0, rejected_total: 0 },
    storage: { file_count: 0 },
    users_usage: [],
    recent_errors: [],
  };
}

function uniqueValues(values) {
  return Array.from(new Set(values.map((value) => String(value || "").trim()).filter(Boolean))).sort();
}

function historyTitle(entry) {
  return entry.title || entry.request?.query || entry.request?.topic || entry.id || "-";
}

function historyRequestLine(request = {}) {
  const parts = [request.query, request.topic, request.sources, request.search_mode, request.year].filter(Boolean);
  return parts.join(" · ") || "-";
}

function formatCounts(counts = {}) {
  const pairs = [
    ["候选", counts.candidates ?? counts.raw ?? counts.raw_count ?? counts.references],
    ["合格", counts.qualified ?? counts.qualified_count],
    ["待复核", counts.needs_review ?? counts.needs_review_count],
    ["过滤", counts.rejected ?? counts.rejected_count],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  return pairs.length ? pairs.map(([label, value]) => `${label} ${value}`).join(" · ") : "-";
}

function buildHistoryInsight(detail = {}) {
  const result = detail.result && typeof detail.result === "object" ? detail.result : {};
  return {
    history_elapsed_seconds: secondsBetween(detail.created_at, detail.updated_at),
    counts: detail.counts || {},
    timings: result.timings || {},
    source_results: result.source_results || result.internal_source_results || {},
    errors: [detail.error || result.error].filter(Boolean),
    warnings: result.diagnostics?.warnings || [],
  };
}

function firstCount(counts = {}, ...keys) {
  for (const key of keys) {
    const value = counts[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return "-";
}

function timingValue(timings = {}, key) {
  const value = timings[key];
  return value === undefined || value === null || value === "" ? null : Number(value);
}

function secondsBetween(start, end) {
  if (!start || !end) return null;
  const startMs = new Date(start).getTime();
  const endMs = new Date(end).getTime();
  if (Number.isNaN(startMs) || Number.isNaN(endMs)) return null;
  return Math.max(0, (endMs - startMs) / 1000);
}

function formatDuration(value) {
  if (value === undefined || value === null || value === "" || Number.isNaN(Number(value))) return "-";
  const seconds = Number(value);
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${rest}s`;
}

function auditActionLabel(action, t) {
  const key = `admin.auditAction.${action || ""}`;
  const label = t(key);
  if (label !== key) return label;
  return String(action || "-").replace(/\./g, " / ");
}

function auditResource(item, t) {
  const type = String(item.resource_type || "").trim();
  return {
    label: auditResourceLabel(type, t),
    id: shortAuditId(item.resource_id, type),
  };
}

function auditResourceLabel(type, t) {
  const key = `admin.auditResource.${type || ""}`;
  const label = t(key);
  return label !== key ? label : type || "-";
}

function auditDetailItems(detail = {}, t) {
  if (!detail || typeof detail !== "object" || Array.isArray(detail)) return [];
  return Object.entries(detail)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => [auditDetailLabel(key, t), auditDetailValue(key, value, t)])
    .filter(([, value]) => value !== "-");
}

function auditDetailLabel(key, t) {
  const labelKey = `admin.auditDetail.${key}`;
  const label = t(labelKey);
  return label !== labelKey ? label : key.replace(/_/g, " ");
}

function auditDetailValue(key, value, t) {
  if (key === "kind") {
    const labelKey = `admin.auditKind.${value}`;
    const label = t(labelKey);
    return label !== labelKey ? label : String(value);
  }
  if (key.endsWith("_user_id") || key === "user_id") return shortAuditId(value, "user");
  if (typeof value === "boolean") return value ? t("admin.yes") : t("admin.no");
  if (Array.isArray(value)) return value.length ? value.join(", ") : "-";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function shortAuditId(value, prefix = "") {
  const raw = String(value || "").trim();
  if (!raw) return "-";
  const cleanPrefix = String(prefix || "").trim();
  const normalized = cleanPrefix && raw.startsWith(cleanPrefix) ? raw.slice(cleanPrefix.length) : raw;
  if (normalized.length <= 16) return normalized;
  return `${normalized.slice(0, 8)}...${normalized.slice(-6)}`;
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}
