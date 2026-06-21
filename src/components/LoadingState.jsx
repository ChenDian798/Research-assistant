export default function LoadingState({ title, message }) {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-squares" aria-hidden="true">
        <span className="loading-square loading-square-large"></span>
        <span className="loading-square loading-square-medium"></span>
        <span className="loading-square loading-square-small"></span>
      </div>
      <div className="loading-copy">
        <h2>{title}</h2>
        <p>{message}</p>
      </div>
    </div>
  );
}
