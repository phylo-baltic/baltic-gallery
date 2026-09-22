from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

METADATA_FILE = Path(__file__).resolve().parents[1] / "source" / "_data" / "content_metadata.json"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    return re.sub(r"-{2,}", "-", value)


def read_metadata() -> dict:
    try:
        data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing content metadata: {METADATA_FILE}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid content metadata: {METADATA_FILE}: {exc}") from exc
    return data


def tag_labels(data: dict) -> dict[str, str]:

    taxonomy_tags = data.get("taxonomy", {}).get("tags", [])
    if not isinstance(taxonomy_tags, list):
        raise SystemExit("Metadata taxonomy.tags must be a list")
    tag_labels = [str(tag).strip() for tag in taxonomy_tags]
    normalized_tags = [slugify(tag) for tag in tag_labels]
    if any(not tag for tag in normalized_tags):
        raise SystemExit("Metadata taxonomy tags must have a non-empty slug")
    labels = {}
    for slug, label in zip(normalized_tags, tag_labels):
        if slug in labels and labels[slug] != label:
            raise SystemExit(f"Metadata taxonomy has conflicting labels for {slug!r}")
        labels[slug] = label
    return labels


def load_metadata(kind: str) -> dict:
    data = read_metadata()
    labels = tag_labels(data)

    section = data.get(kind)
    if not isinstance(section, dict):
        raise SystemExit(f"Metadata section {kind!r} must be an object")
    # Keep the taxonomy spelling as the canonical display label. Item tags are
    # matched by slug so, for example, both "MERS-CoV" and "mers-cov" resolve
    # to the taxonomy label "MERS-CoV".
    section["tag_labels"] = labels
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
        tag_slug = slugify(str(tag))
        if tag_slug:
            if tag_slug not in section["tag_labels"]:
                raise SystemExit(f"Unknown content tag {tag_slug!r}; add it to the taxonomy first")
            tag_label = section["tag_labels"][tag_slug]
            if tag_label not in result:
                result.append(tag_label)
    return result


def tag_chips(tags: list[str], href_prefix: str = "../tags/") -> str:
    return "".join(
        f'<a class="content-tag content-tag--{slugify(tag)}" href="{href_prefix}{slugify(tag)}.html">{escape(tag)}</a>'
        for tag in tags
    )
