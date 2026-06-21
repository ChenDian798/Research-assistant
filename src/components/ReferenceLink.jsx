import { linkParts } from "../lib/formatters.js";

export default function ReferenceLink({ title, source, uploadedFilename, t }) {
  const parts = linkParts(title, source, uploadedFilename);
  const sourceLine = translatedSourceLine(parts.sourceLine, t);
  return (
    <>
      {parts.href ? (
        <a href={parts.href} target="_blank" rel="noreferrer">{parts.title}</a>
      ) : (
        parts.title
      )}
      {sourceLine ? <small className="source-filename">{sourceLine}</small> : null}
    </>
  );
}

function translatedSourceLine(sourceLine, t) {
  if (!sourceLine || !t) return sourceLine;
  if (sourceLine.startsWith("文件：")) return `${t("reference.file")}: ${sourceLine.slice(3)}`;
  if (sourceLine.startsWith("来源：")) return `${t("reference.source")}: ${sourceLine.slice(3)}`;
  return sourceLine;
}
