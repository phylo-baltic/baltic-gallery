#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from pathlib import Path

from pybtex.backends.html import Backend
from pybtex.database import BibliographyData, Entry, parse_file
from pybtex.richtext import Text
from pybtex.style.formatting import toplevel
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.formatting.unsrt import pages
from pybtex.style.template import (
    field,
    href,
    join,
    optional,
    sentence,
    tag,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BIB_FILE = PROJECT_ROOT / "source" / "baltic-uses" / "baltic-uses.bib"
USES_DOCS = PROJECT_ROOT / "source" / "uses"

PREPRINT_SOURCES = {
    "biorxiv",
    "medrxiv",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalized_doi(entry: Entry) -> str:
    doi = entry.fields.get("doi", "").strip()
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)


def entry_link(entry: Entry) -> str:
    doi = normalized_doi(entry)
    url = entry.fields.get("url", "").strip()
    link = f"https://doi.org/{doi}" if doi else url
    return html.escape(link, quote=True)


def is_preprint_source(source: str) -> bool:
    return source.strip().lower() in PREPRINT_SOURCES


def preprint_identifier(entry: Entry) -> str:
    source = entry.fields.get("publisher", entry.fields.get("institution", ""))
    if not is_preprint_source(source):
        return ""

    doi = normalized_doi(entry)
    return doi.rstrip("/").split("/")[-1] if doi else ""


def entry_year(entry: Entry) -> int:
    match = re.search(r"\d{4}", entry.fields.get("year", ""))
    return int(match.group(0)) if match else 0


def normalize_bibliography(data: BibliographyData) -> None:
    """Normalize aliases that Pybtex does not interpret as standard BibTeX fields."""
    for entry in data.entries.values():
        if "number" not in entry.fields and "issue" in entry.fields:
            entry.fields["number"] = entry.fields["issue"]


class WebsiteBackend(Backend):
    """Render BibTeX capitalization guards without presentational markup."""

    def format_protected(self, text: str) -> str:
        return self.format_str(text)


class WebsiteStyle(UnsrtStyle):
    """Compact, linked citations using Pybtex's standard article formatting."""

    default_name_style = "plain"
    default_sorting_style = "none"

    def __init__(self) -> None:
        super().__init__(abbreviate_names=True)

    def format_title(self, entry: Entry, which_field: str, as_sentence: bool = True):
        title = field(which_field)
        link = entry_link(entry)
        if link:
            title = href(link)[title]
        return sentence[title] if as_sentence else title

    def format_web_refs(self, entry: Entry) -> Text:
        # Titles already link to the DOI (or URL), so repeating long web references
        # makes the publication list harder to scan.
        return Text()

    def get_article_template(self, entry: Entry):
        has_volume = bool(entry.fields.get("volume"))
        has_number = bool(entry.fields.get("number"))
        has_pages = bool(entry.fields.get("pages"))

        publication_details = None
        if has_volume:
            publication_details = join[
                field("volume"), optional["(", field("number"), ")"]
            ]
        elif has_number:
            publication_details = join["(", field("number"), ")"]

        if has_pages:
            publication_details = (
                join[publication_details, ":", pages]
                if publication_details is not None
                else pages
            )

        return toplevel[
            self.format_names("author"),
            self.format_title(entry, "title"),
            sentence[
                tag("em")[field("journal")],
                optional[publication_details] if publication_details is not None else "",
                field("year"),
            ],
        ]

    def get_misc_template(self, entry: Entry):
        return self._get_repository_template(entry, "publisher")

    def get_techreport_template(self, entry: Entry):
        return self._get_repository_template(entry, "institution")

    def _get_repository_template(self, entry: Entry, source_field: str):
        identifier = preprint_identifier(entry)
        source_details = join(sep=", ")[
            optional[tag("em")[field(source_field)]],
            optional[identifier],
            field("year"),
        ]
        return toplevel[
            optional[self.format_names("author")],
            optional[self.format_title(entry, "title")],
            sentence[source_details],
        ]


def ordered_citation_keys(data: BibliographyData) -> list[str]:
    source_order = {key: index for index, key in enumerate(data.entries)}
    return sorted(
        data.entries,
        key=lambda key: (entry_year(data.entries[key]), source_order[key]),
        reverse=True,
    )


def citation_html_by_key(data: BibliographyData) -> dict[str, str]:
    style = WebsiteStyle()
    backend = WebsiteBackend()
    keys = ordered_citation_keys(data)
    formatted = style.format_bibliography(data, citations=keys)
    return {entry.key: entry.text.render(backend) for entry in formatted}


def write_uses_page(data: BibliographyData) -> None:
    ensure_dir(USES_DOCS)

    ordered_keys = ordered_citation_keys(data)
    citations = citation_html_by_key(data)
    total = len(ordered_keys)
    items = "\n".join(
        f'<li value="{total - index}">{citations[key]}</li>'
        for index, key in enumerate(ordered_keys)
    )

    rst = f"""\
As seen in
==========

.. raw:: html

   <ol class="uses-list">
{indent_html(items, 6)}
   </ol>
"""
    (USES_DOCS / "index.rst").write_text(rst, encoding="utf-8")


def indent_html(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def main() -> None:
    if not BIB_FILE.exists():
        raise SystemExit(f"Missing bibliography file: {BIB_FILE}")

    data = parse_file(BIB_FILE, bib_format="bibtex")
    normalize_bibliography(data)
    write_uses_page(data)
    print(f"Generated {len(data.entries)} use records with Pybtex.")
    print(f"- Landing: {USES_DOCS / 'index.rst'}")


if __name__ == "__main__":
    main()
