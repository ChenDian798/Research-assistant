from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from .pubmed_search import fetch_pubmed_records


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s,;，；]+", re.IGNORECASE)
ARXIV_PATTERN = re.compile(
    r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?([a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
PMID_PATTERN = re.compile(r"\b(?:PMID|PubMed\s+ID)\s*:?\s*(\d{6,9})\b", re.IGNORECASE)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _user_agent() -> str:
    contact = os.getenv("RESEARCH_AGENT_CONTACT_EMAIL", "admin@example.com").strip()
    return f"ResearchAgent/0.1 (mailto:{contact})"


def enrich_references_with_doi_metadata(references: list[dict]) -> list[dict]:
    enriched = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        item = dict(reference)
        doi = extract_doi(item)
        if doi:
            metadata = fetch_crossref_metadata(doi)
            if metadata:
                item.update(metadata)
                item["source"] = f"https://doi.org/{doi}"
                item["relevance"] = metadata_relevance(metadata, item.get("relevance", ""))
        elif arxiv_id := extract_arxiv_id(item):
            metadata = fetch_arxiv_metadata(arxiv_id)
            if metadata:
                item.update(metadata)
                item["source"] = f"https://arxiv.org/abs/{metadata['arxiv_id']}"
                item["relevance"] = metadata_relevance(metadata, item.get("relevance", ""))
        elif pmid := extract_pmid(item):
            metadata = fetch_pubmed_metadata(pmid)
            if metadata:
                item.update(metadata)
                item["source"] = f"https://pubmed.ncbi.nlm.nih.gov/{metadata['pmid']}/"
                item["relevance"] = metadata_relevance(metadata, item.get("relevance", ""))
        elif url := extract_url(item):
            metadata = fetch_webpage_metadata(url)
            if metadata:
                item.update(metadata)
                item["relevance"] = metadata_relevance(metadata, item.get("relevance", ""))
        enriched.append(item)
    return enriched


def extract_doi(reference: dict) -> str:
    text = " ".join(
        str(reference.get(key, "") or "")
        for key in ("doi", "title", "source", "relevance", "bibliographic_identity")
    )
    match = DOI_PATTERN.search(text)
    if not match:
        return ""
    return match.group(0).strip().rstrip(".)]}")


def extract_arxiv_id(reference: dict) -> str:
    text = " ".join(
        str(reference.get(key, "") or "")
        for key in ("arxiv_id", "id", "title", "source", "relevance", "bibliographic_identity")
    )
    text = unquote(text)
    for token in re.split(r"\s+", text):
        cleaned = token.strip().rstrip(".)]}")
        parsed = urlparse(cleaned)
        if parsed.netloc.lower().endswith("arxiv.org"):
            candidate = parsed.path.strip("/")
            candidate = re.sub(r"^(abs|pdf)/", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
            match = ARXIV_PATTERN.search(candidate)
            if match:
                return match.group(1)
        if parsed.scheme or parsed.netloc:
            continue
        if re.fullmatch(r"(?:arxiv:)?(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", cleaned, flags=re.IGNORECASE):
            match = ARXIV_PATTERN.search(cleaned)
            if match:
                return match.group(1)

    marker_match = re.search(
        r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)([a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
        text,
        flags=re.IGNORECASE,
    )
    return marker_match.group(1) if marker_match else ""


def extract_pmid(reference: dict) -> str:
    explicit = str(reference.get("pmid", "") or "").strip()
    if re.fullmatch(r"\d{6,9}", explicit):
        return explicit

    text = " ".join(
        str(reference.get(key, "") or "")
        for key in ("title", "source", "relevance", "bibliographic_identity")
    )
    text = unquote(text)
    for token in re.split(r"\s+", text):
        parsed = urlparse(token.strip().rstrip(".)]}"))
        host = parsed.netloc.lower()
        if host.endswith("pubmed.ncbi.nlm.nih.gov"):
            match = re.search(r"/(\d{6,9})(?:/|$)", parsed.path)
            if match:
                return match.group(1)
        if host.endswith("ncbi.nlm.nih.gov") and "/pubmed/" in parsed.path.lower():
            match = re.search(r"/pubmed/(\d{6,9})(?:/|$)", parsed.path, flags=re.IGNORECASE)
            if match:
                return match.group(1)
    match = PMID_PATTERN.search(text)
    return match.group(1) if match else ""


def extract_url(reference: dict) -> str:
    text = " ".join(
        str(reference.get(key, "") or "")
        for key in ("title", "source", "relevance")
    )
    match = re.search(r"https?://[^\s,;，；]+", text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(0).strip().rstrip(".)]}")


def fetch_crossref_metadata(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": _user_agent(),
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return {}

    message = payload.get("message", {})
    if not isinstance(message, dict):
        return {}

    title = first_text(message.get("title"))
    if not title:
        title = f"DOI: {doi}"

    authors = format_authors(message.get("author", []))
    year = published_year(message)
    journal = first_text(message.get("container-title"))
    abstract = clean_abstract(str(message.get("abstract", "") or ""))

    return {
        "doi": doi,
        "title": title,
        "source": f"https://doi.org/{doi}",
        "authors": authors,
        "year": str(year) if year else "",
        "journal": journal,
        "abstract": abstract,
    }


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={quote(arxiv_id, safe='/')}"
    request = Request(
        url,
        headers={
            "Accept": "application/atom+xml",
            "User-Agent": _user_agent(),
        },
    )
    root = None
    for _ in range(3):
        try:
            with urlopen(request, timeout=12) as response:
                payload = response.read()
            root = ET.fromstring(payload)
            break
        except IncompleteRead as error:
            try:
                root = ET.fromstring(error.partial)
                break
            except ET.ParseError:
                continue
        except (HTTPError, URLError, TimeoutError, ET.ParseError, OSError):
            continue

    if root is None:
        return {}

    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        return {}

    title = clean_abstract(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
    abstract = clean_abstract(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
    author_nodes = entry.findall("atom:author", ATOM_NS)
    authors = []
    for author in author_nodes[:6]:
        name = clean_abstract(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        if name:
            authors.append(name)
    if len(author_nodes) > 6:
        authors.append("et al.")

    canonical_id = arxiv_id
    id_text = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    match = ARXIV_PATTERN.search(id_text)
    if match:
        canonical_id = match.group(1)

    return {
        "arxiv_id": canonical_id,
        "title": title or f"arXiv: {canonical_id}",
        "source": f"https://arxiv.org/abs/{canonical_id}",
        "authors": ", ".join(authors),
        "year": published[:4] if published else "",
        "journal": "arXiv",
        "abstract": abstract,
    }


def fetch_pubmed_metadata(pmid: str) -> dict:
    try:
        records = fetch_pubmed_records([pmid])
    except Exception:
        return {}
    if not records:
        return {}
    record = records[0]
    authors = record.get("authors", [])
    if isinstance(authors, list):
        authors_text = ", ".join(str(author) for author in authors[:6] if author)
        if len(authors) > 6:
            authors_text = f"{authors_text}, et al." if authors_text else "et al."
    else:
        authors_text = str(authors or "")
    return {
        "pmid": str(record.get("pmid") or pmid),
        "doi": str(record.get("doi") or ""),
        "title": str(record.get("title") or f"PubMed PMID: {pmid}"),
        "source": str(record.get("abs_url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
        "authors": authors_text,
        "year": str(record.get("published") or ""),
        "journal": str(record.get("journal") or "PubMed"),
        "abstract": str(record.get("abstract") or ""),
    }


def fetch_webpage_metadata(url: str) -> dict:
    inferred_doi = infer_doi_from_known_url(url)
    if inferred_doi:
        metadata = fetch_crossref_metadata(inferred_doi)
        if metadata:
            return metadata

    html = fetch_html(url)
    if not html:
        return {}

    meta = extract_html_meta(html)
    doi = first_available(
        meta,
        "citation_doi",
        "dc.identifier",
        "dc.identifier.doi",
        "prism.doi",
        "doi",
    )
    if doi:
        doi = doi.replace("doi:", "").replace("https://doi.org/", "").strip()
        metadata = fetch_crossref_metadata(doi)
        if metadata:
            return metadata

    title = first_available(meta, "citation_title", "dc.title", "og:title", "twitter:title")
    authors = "; ".join(meta.get("citation_author", [])[:6])
    year = first_year(first_available(meta, "citation_publication_date", "dc.date", "article:published_time"))
    journal = first_available(meta, "citation_journal_title", "dc.source", "og:site_name")
    abstract = first_available(meta, "citation_abstract", "dc.description", "description", "og:description")

    if not any((title, authors, year, journal, abstract)):
        return {}

    return {
        "title": title or url,
        "source": url,
        "authors": authors,
        "year": year,
        "journal": journal,
        "abstract": clean_abstract(abstract),
    }


def infer_doi_from_known_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    if host.endswith("nature.com") and len(path_parts) >= 2 and path_parts[0] == "articles":
        article_id = path_parts[1].removesuffix(".pdf")
        if article_id:
            return f"10.1038/{article_id}"
    return ""


def fetch_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": _user_agent(),
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            content_type = response.headers.get_content_charset() or "utf-8"
            return response.read(1_000_000).decode(content_type, errors="replace")
    except (HTTPError, URLError, TimeoutError, IncompleteRead, OSError, UnicodeError):
        return ""


def extract_html_meta(html: str) -> dict[str, list[str]]:
    parser = MetaTagParser()
    parser.feed(html)
    return parser.meta


class MetaTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        attr_map = {key.lower(): value or "" for key, value in attrs}
        name = (attr_map.get("name") or attr_map.get("property") or "").strip().lower()
        content = clean_abstract(attr_map.get("content", ""))
        if name and content:
            self.meta.setdefault(name, []).append(content)


def first_available(values: dict[str, list[str]], *keys: str) -> str:
    for key in keys:
        items = values.get(key.lower(), [])
        if items:
            return items[0]
    return ""


def first_year(value: str) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", value or "")
    return match.group(0) if match else ""


def metadata_relevance(metadata: dict, fallback: str) -> str:
    parts = []
    if metadata.get("authors"):
        parts.append(f"作者: {metadata['authors']}")
    if metadata.get("year"):
        parts.append(f"年份: {metadata['year']}")
    if metadata.get("journal"):
        parts.append(f"来源: {metadata['journal']}")
    if metadata.get("abstract"):
        parts.append(f"摘要: {metadata['abstract'][:700]}")
    if fallback:
        parts.append(str(fallback))
    return "；".join(parts) or "用户在文献助手中主动提交，已解析 DOI 元数据。"


def first_text(value) -> str:
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def format_authors(authors) -> str:
    if not isinstance(authors, list):
        return ""
    names = []
    for author in authors[:6]:
        if not isinstance(author, dict):
            continue
        given = str(author.get("given", "") or "").strip()
        family = str(author.get("family", "") or "").strip()
        name = " ".join(part for part in (given, family) if part)
        if name:
            names.append(name)
    if len(authors) > 6:
        names.append("et al.")
    return ", ".join(names)


def published_year(message: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        value = message.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
        ):
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def clean_abstract(value: str) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
