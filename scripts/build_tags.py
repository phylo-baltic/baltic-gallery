#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "source"
TAGS_DOCS = SOURCE / "tags"
ITEM_FILES = [SOURCE / "examples" / ".content_items.json", SOURCE / "tutorials" / ".content_items.json"]


def main() -> None:
    items = []
    for item_file in ITEM_FILES:
        if item_file.exists():
            section = json.loads(item_file.read_text(encoding="utf-8"))
            prefix = item_file.parent.name
            items.extend({**item, "href": f"../{prefix}/{item['href']}"} for item in section)

    by_tag = defaultdict(list)
    for item in items:
        for tag in item.get("tags", []):
            by_tag[tag].append(item)

    TAGS_DOCS.mkdir(parents=True, exist_ok=True)
    sections = []
    for tag in sorted(by_tag):
        links = "\n".join(
            f'   * `{escape(item["title"])} <{escape(item["href"], quote=True)}>`__ ({item["type"]})'
            for item in sorted(by_tag[tag], key=lambda value: value["title"].lower())
        )
        title = tag.replace("-", " ").title()
        sections.append(f"{title}\n{'-' * len(title)}\n\n{links}\n")

    body = "\n".join(sections) or "No tagged content yet.\n"
    rst = f"Tags\n====\n\nBrowse examples and tutorials by tag.\n\n{body}"
    (TAGS_DOCS / "index.rst").write_text(rst, encoding="utf-8")
    print(f"Generated {len(by_tag)} tag pages/sections from {len(items)} content items.")


if __name__ == "__main__":
    main()
