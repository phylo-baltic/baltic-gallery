#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "source"
TAGS_DOCS = SOURCE / "tags"
MANIFEST_FILE = TAGS_DOCS / ".build_tags_manifest.json"
ITEM_FILES = [SOURCE / "examples" / ".content_items.json", SOURCE / "tutorials" / ".content_items.json"]


def clean_previous_outputs() -> None:
    if not MANIFEST_FILE.exists():
        return
    try:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for relative_path in manifest.get("generated", []):
        path = PROJECT_ROOT / relative_path
        if path.is_file():
            path.unlink()


def card(item: dict) -> str:
    image = item.get("image") or "_static/no_image.png"
    return f'''
<div class="gallery-card">
  <div class="gallery-card__imgwrap">
        <a class="gallery-card__image-link" href="{escape(item["href"], quote=True)}">
            <img src="../{escape(image, quote=True)}" alt="{escape(item["title"], quote=True)}" loading="lazy">
        </a>
    <div class="gallery-card__overlay">
      <a class="gallery-card__title" href="{escape(item["href"], quote=True)}">{escape(item["title"])}</a>
    </div>
  </div>
</div>
'''.strip()


def main() -> None:
    clean_previous_outputs()
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
    generated = []
    index_links = []
    for tag in sorted(by_tag):
        tag_title = tag.replace("-", " ").title()
        tag_page = TAGS_DOCS / f"{tag}.rst"
        cards = "\n".join(card(item) for item in sorted(by_tag[tag], key=lambda value: value["title"].lower()))
        indented_cards = "\n".join(f"   {line}" for line in cards.splitlines())
        rst = f"{tag_title}\n{'=' * len(tag_title)}\n\n.. raw:: html\n\n{indented_cards}\n"
        tag_page.write_text(rst, encoding="utf-8")
        generated.append(tag_page.relative_to(PROJECT_ROOT).as_posix())
        index_links.append(f"   * `{tag_title} <{tag}.html>`__")

    index_path = TAGS_DOCS / "index.rst"
    toc = "\n.. toctree::\n   :hidden:\n\n" + "\n".join(f"   {tag}" for tag in sorted(by_tag)) + "\n"
    index_path.write_text("Tags\n====\n\nBrowse the generated gallery pages by tag.\n\n" + "\n".join(index_links) + toc, encoding="utf-8")
    generated.append(index_path.relative_to(PROJECT_ROOT).as_posix())
    MANIFEST_FILE.write_text(json.dumps({"generated": sorted(generated)}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(by_tag)} tag gallery pages from {len(items)} content items.")


if __name__ == "__main__":
    main()
