from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedSearchError(RuntimeError):
    pass


def search_pubmed(
    query: str,
    *,
    max_results: int = 20,
    start_date: str = "",
    end_date: str = "",
) -> list[dict]:
    max_results = max(1, min(int(max_results or 20), 50))
    ids = pubmed_search_ids(query, max_results=max_results, start_date=start_date, end_date=end_date)
    if not ids:
        return []
    return fetch_pubmed_records(ids)


def pubmed_search_ids(query: str, *, max_results: int, start_date: str, end_date: str) -> list[str]:
    params = eutils_base_params()
    params.update(
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        }
    )
    if start_date or end_date:
        params.update(
            {
                "datetype": "pdat",
                "mindate": start_date or "1900/01/01",
                "maxdate": end_date or "3000/12/31",
            }
        )
    payload = fetch_json(f"{EUTILS_BASE}/esearch.fcgi", params)
    return [str(item) for item in payload.get("esearchresult", {}).get("idlist", [])]


def fetch_pubmed_records(ids: list[str]) -> list[dict]:
    params = eutils_base_params()
    params.update({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
    xml_text = fetch_text(f"{EUTILS_BASE}/efetch.fcgi", params)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise PubMedSearchError("PubMed returned invalid XML.") from error
    records = []
    for article in root.findall(".//PubmedArticle"):
        records.append(normalize_pubmed_article(article))
    return [record for record in records if record.get("title")]


def normalize_pubmed_article(article: ET.Element) -> dict:
    pmid = text(article.find(".//PMID"))
    title = clean_text(" ".join(article.findtext(".//ArticleTitle", default="").split()))
    abstract_parts = [clean_text("".join(node.itertext())) for node in article.findall(".//Abstract/AbstractText")]
    abstract = " ".join(part for part in abstract_parts if part)
    journal = clean_text(article.findtext(".//Journal/Title", default=""))
    year = first_year(article)
    doi = ""
    for node in article.findall(".//ArticleId"):
        if (node.attrib.get("IdType") or "").lower() == "doi":
            doi = text(node)
            break
    authors = []
    for author in article.findall(".//AuthorList/Author")[:12]:
        collective = clean_text(author.findtext("CollectiveName", default=""))
        if collective:
            authors.append(collective)
            continue
        fore = clean_text(author.findtext("ForeName", default=""))
        last = clean_text(author.findtext("LastName", default=""))
        name = " ".join(part for part in (fore, last) if part)
        if name:
            authors.append(name)
    return {
        "id": doi or pmid,
        "title": title or f"PubMed PMID: {pmid}",
        "authors": authors,
        "abstract": abstract,
        "published": year,
        "updated": "",
        "categories": ["PubMed"],
        "abs_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else ""),
        "pdf_url": "",
        "source": "PubMed",
        "doi": doi,
        "journal": journal,
        "pmid": pmid,
    }


def eutils_base_params() -> dict:
    params = {"tool": "ResearchAgentLiteratureAnalysis"}
    if os.getenv("NCBI_EMAIL"):
        params["email"] = os.getenv("NCBI_EMAIL")
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
    return params


def fetch_json(url: str, params: dict) -> dict:
    try:
        return json.loads(fetch_text(url, params))
    except json.JSONDecodeError as error:
        raise PubMedSearchError("PubMed returned invalid JSON.") from error


def fetch_text(url: str, params: dict) -> str:
    request = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "ResearchAgent-LiteratureAnalysis/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise PubMedSearchError(f"PubMed search failed: {error}") from error


def text(node: ET.Element | None) -> str:
    return clean_text("".join(node.itertext())) if node is not None else ""


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def first_year(article: ET.Element) -> str:
    for path in [".//ArticleDate/Year", ".//JournalIssue/PubDate/Year", ".//PubMedPubDate/Year"]:
        value = clean_text(article.findtext(path, default=""))
        if re.fullmatch(r"\d{4}", value):
            return value
    medline = clean_text(article.findtext(".//JournalIssue/PubDate/MedlineDate", default=""))
    match = re.search(r"\b(19|20)\d{2}\b", medline)
    return match.group(0) if match else ""
