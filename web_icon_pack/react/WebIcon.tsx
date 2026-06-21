import React from "react";

export const webIcons = {
  "search-flow/analysis": "svg/search-flow/analysis.svg",
  "search-flow/checklist": "svg/search-flow/checklist.svg",
  "search-flow/database": "svg/search-flow/database.svg",
  "search-flow/filter": "svg/search-flow/filter.svg",
  "search-flow/link": "svg/search-flow/link.svg",
  "search-flow/search": "svg/search-flow/search.svg",
  "search-flow/start": "svg/search-flow/start.svg",
  "search-flow/step-1-search": "svg/search-flow/step-1-search.svg",
  "search-flow/step-2-select": "svg/search-flow/step-2-select.svg",
  "search-flow/step-3-confirm": "svg/search-flow/step-3-confirm.svg",
  "search-flow/step-4-analyze": "svg/search-flow/step-4-analyze.svg",
  "search-flow/upload": "svg/search-flow/upload.svg",
  "research-assistant/analysis-table": "svg/research-assistant/analysis-table.svg",
  "research-assistant/checklist": "svg/research-assistant/checklist.svg",
  "research-assistant/document-search": "svg/research-assistant/document-search.svg",
  "research-assistant/doi-link": "svg/research-assistant/doi-link.svg",
  "research-assistant/empty-state-paper": "svg/research-assistant/empty-state-paper.svg",
  "research-assistant/pubmed-dna": "svg/research-assistant/pubmed-dna.svg",
  "research-assistant/research-assistant-logo-mark": "svg/research-assistant/research-assistant-logo-mark.svg",
  "research-assistant/upload-cloud": "svg/research-assistant/upload-cloud.svg",
  "document-magnifier/document-magnifier-hero": "svg/document-magnifier/document-magnifier-hero.svg",
} as const;

export type WebIconName = keyof typeof webIcons;

type WebIconProps = Omit<React.ImgHTMLAttributes<HTMLImageElement>, "src" | "width" | "height"> & {
  name: WebIconName;
  size?: number | string;
  basePath?: string;
};

function joinPath(basePath: string, iconPath: string) {
  return [basePath.replace(/\/$/, ""), iconPath].filter(Boolean).join("/");
}

export function WebIcon({
  name,
  size = 40,
  basePath = "",
  className,
  alt = "",
  loading = "lazy",
  ...imageProps
}: WebIconProps) {
  return (
    <img
      {...imageProps}
      className={["web-icon", className].filter(Boolean).join(" ")}
      src={joinPath(basePath, webIcons[name])}
      width={size}
      height={size}
      alt={alt}
      aria-hidden={alt === "" ? true : imageProps["aria-hidden"]}
      loading={loading}
    />
  );
}
