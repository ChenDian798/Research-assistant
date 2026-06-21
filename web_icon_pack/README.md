# Web Icon Pack

Merged icon pack for literature search, research-assistant web UI, and document magnifier illustrations.

## Contents

- `svg/search-flow/`: 12 workflow icons from `search_flow_icon_pack`.
- `svg/research-assistant/`: 8 research UI icons from `research_assistant_icon_pack`.
- `svg/document-magnifier/`: 1 hero illustration from `document_magnifier_illustration_pack`.
- `react/WebIcon.tsx`: React image wrapper with typed icon names.
- `react/DocumentMagnifierHero.tsx`: original editable React SVG component for the hero illustration.
- `web-icon-theme.css`: shared CSS variables and preview/card styles.
- `manifest.json`: machine-readable icon index.
- `preview.html`: local browser preview for all icons.

The source folders are preserved. This package removes the duplicated nested
`search_flow_icon_pack/search_flow_icon_pack/svg` copy and keeps same-name icons
in separate categories, so files such as `checklist.svg` do not overwrite each
other.

## Direct SVG Usage

```html
<img src="./svg/search-flow/search.svg" alt="Search" width="40" height="40" />
<img src="./svg/research-assistant/document-search.svg" alt="Document search" width="40" height="40" />
<img src="./svg/document-magnifier/document-magnifier-hero.svg" alt="Document magnifier" width="320" />
```

## React Usage

```tsx
import { WebIcon } from "./react/WebIcon";
import "./web-icon-theme.css";

export function Example() {
  return (
    <button className="web-icon-button" type="button">
      <WebIcon name="search-flow/search" size={24} alt="" />
      Search literature
    </button>
  );
}
```

If the SVG folder is served from another public path, pass `basePath`:

```tsx
<WebIcon name="research-assistant/upload-cloud" basePath="/assets/web_icon_pack" />
```

For a larger editable hero illustration, import the dedicated React component:

```tsx
import { DocumentMagnifierHero } from "./react/DocumentMagnifierHero";

export function HeroArt() {
  return <DocumentMagnifierHero width="100%" height="auto" />;
}
```

## Icon Names

### Search Flow

- `search-flow/analysis`
- `search-flow/checklist`
- `search-flow/database`
- `search-flow/filter`
- `search-flow/link`
- `search-flow/search`
- `search-flow/start`
- `search-flow/step-1-search`
- `search-flow/step-2-select`
- `search-flow/step-3-confirm`
- `search-flow/step-4-analyze`
- `search-flow/upload`

### Research Assistant

- `research-assistant/analysis-table`
- `research-assistant/checklist`
- `research-assistant/document-search`
- `research-assistant/doi-link`
- `research-assistant/empty-state-paper`
- `research-assistant/pubmed-dna`
- `research-assistant/research-assistant-logo-mark`
- `research-assistant/upload-cloud`

### Document Magnifier

- `document-magnifier/document-magnifier-hero`

## Notes

The research-assistant SVGs use CSS variables such as `--ra-blue`,
`--ra-teal`, and `--ra-paper`. Override them globally or on a wrapper element
to recolor those icons.
