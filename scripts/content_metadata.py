from __future__ import annotations

import json
import re
from pathlib import Path

METADATA_FILE = Path(__file__).resolve().parents[1] / "source" / "_data" / "content_metadata.json"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    return re.sub(r"-{2,}", "-", value)


def load_metadata(kind: str) -> dict:
    try:
        data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing content metadata: {METADATA_FILE}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid content metadata: {METADATA_FILE}: {exc}") from exc

    taxonomy_tags = data.get("taxonomy", {}).get("tags", [])
    if not isinstance(taxonomy_tags, list):
        raise SystemExit("Metadata taxonomy.tags must be a list")
    normalized_tags = [slugify(str(tag)) for tag in taxonomy_tags]
    if len(normalized_tags) != len(set(normalized_tags)):
        raise SystemExit("Metadata taxonomy contains duplicate tag slugs")

    section = data.get(kind)
    if not isinstance(section, dict):
        raise SystemExit(f"Metadata section {kind!r} must be an object")
    section["allowed_tags"] = set(normalized_tags)
    return section


def tags_for(section: dict, source_key: str, category: str | None = None) -> list[str]:
    item_tags = section.get("items", {}).get(source_key)
    if item_tags is not None:
        tags = list(item_tags)
    else:
        tags = list(section.get("defaults", []))
        if category:
            tags.extend(section.get("categories", {}).get(category, []))
    result = []
    for tag in tags:
        tag = slugify(str(tag))
        if tag and tag not in result:
            if tag not in section["allowed_tags"]:
                raise SystemExit(f"Unknown content tag {tag!r}; add it to the taxonomy first")
            result.append(tag)
    return result


def tag_chips(tags: list[str], href_prefix: str = "../tags/") -> str:
    return "".join(
        f'<a class="content-tag content-tag--{slugify(tag)}" href="{href_prefix}{slugify(tag)}.html">{tag.replace("-", " ")}</a>'
        for tag in tags
    )
