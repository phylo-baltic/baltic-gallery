#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path

from content_metadata import load_metadata, tag_chips, tags_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TUTORIALS_SRC = PROJECT_ROOT / "source" / "baltic-tutorials"
SPHINX_SOURCE = PROJECT_ROOT / "source"
TUTORIALS_DOCS = SPHINX_SOURCE / "tutorials"
STATIC_TUTORIALS = SPHINX_SOURCE / "_static" / "tutorials"
DESC_FILE = SPHINX_SOURCE / "_data" / "tutorials_description.txt"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SOURCE_EXTS = {".py", ".ipynb"}
MANIFEST_FILE = TUTORIALS_DOCS / ".build_tutorials_manifest.json"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()


def safe_read_text(p: Path, max_chars: int = 200_000) -> str:
    try:
        txt = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = p.read_text(encoding="latin-1")
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "\n# ... truncated ...\n"
    return txt


def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def html_src(static_path: str) -> str:
    return f"../{static_path}"


def rst_static_path(static_path: str) -> str:
    return f"../{static_path}"


def notebook_code(path: Path, max_chars: int = 200_000) -> str:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return f"# Could not read notebook source: {exc}\n"

    code_cells: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            code_cells.append(source.rstrip())

    code = "\n\n# %%\n\n".join(code_cells)
    if not code:
        code = "# This notebook does not contain Python code cells.\n"
    if len(code) > max_chars:
        code = code[:max_chars] + "\n# ... truncated ...\n"
    return code


def notebook_title(path: Path) -> str | None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        for line in source.splitlines():
            title = line.strip().removeprefix("#").strip()
            if line.lstrip().startswith("#") and title:
                return title
    return None


def read_source_code(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return notebook_code(path)
    return safe_read_text(path)


def title_from_source(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        title = notebook_title(path)
        if title:
            return title
    return path.stem.replace("-", " ").replace("_", " ")


def find_matching_image(source: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = source.with_suffix(ext)
        if candidate.exists():
            return candidate

    source_stem = source.stem.lower()
    images = [
        p for p in TUTORIALS_SRC.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem.lower() == source_stem
    ]
    return sorted(images, key=lambda p: p.name.lower())[0] if images else None


def copy_static_file(source: Path) -> str:
    ensure_dir(STATIC_TUTORIALS)
    dest = STATIC_TUTORIALS / source.name
    shutil.copy2(source, dest)
    return f"_static/tutorials/{source.name}"


def unique_doc_slug(base_slug: str, used: set[str]) -> str:
    candidate = base_slug
    counter = 2
    while candidate in used:
        candidate = f"{base_slug}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


@dataclass
class TutorialItem:
    title: str
    source_name: str
    source_suffix: str
    img_web: str | None
    source_web: str
    rst_doc: str
    html_href: str
    source_path: Path
    tags: list[str]


def collect_items() -> list[TutorialItem]:
    if not TUTORIALS_SRC.exists():
        raise SystemExit(
            f"Missing input folder: {TUTORIALS_SRC}\n"
            "Create it and copy the tutorial notebooks/images there first."
        )

    sources = [
        p for p in TUTORIALS_SRC.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCE_EXTS
    ]
    sources.sort(key=lambda p: p.name.lower())

    items: list[TutorialItem] = []
    used_doc_slugs: set[str] = set()
    metadata = load_metadata("tutorials")

    for source in sources:
        doc_slug = unique_doc_slug(slugify(source.stem), used_doc_slugs)
        img = find_matching_image(source)

        items.append(
            TutorialItem(
                title=title_from_source(source),
                source_name=source.stem,
                source_suffix=source.suffix.lower(),
                img_web=copy_static_file(img) if img else None,
                source_web=copy_static_file(source),
                rst_doc=doc_slug,
                html_href=f"{doc_slug}.html",
                source_path=source,
                tags=tags_for(metadata, source.name),
            )
        )

    return items


def write_tutorial_page(item: TutorialItem) -> None:
    ensure_dir(TUTORIALS_DOCS)

    hero = item.img_web or "_static/no_image.png"
    source_kind = "Notebook" if item.source_suffix == ".ipynb" else "Python source"
    download_label = escape(f"Download {source_kind.lower()}")
    code = read_source_code(item.source_path)
    indented_code = indent(code.rstrip() + "\n", 3)
    underline = "=" * len(item.title)

    rst = f"""\
{item.title}
{underline}

.. image:: {rst_static_path(hero)}
   :alt: {item.source_name}
   :class: example-detail__hero-img

.. raw:: html

   <p class="example-detail__source-link">
     <a href="{html_src(item.source_web)}" download>{download_label}</a>
   </p>

Code
----

.. code-block:: python

{indented_code}
"""
    (TUTORIALS_DOCS / f"{item.rst_doc}.rst").write_text(rst, encoding="utf-8")


def write_tutorials_landing(items: list[TutorialItem]) -> None:
    ensure_dir(TUTORIALS_DOCS)

    desc = ""
    if DESC_FILE.exists():
        desc = read_text(DESC_FILE)

    cards: list[str] = []
    for item in items:
        img = item.img_web or "_static/no_image.png"
        cards.append(
            f"""
<a class="gallery-card" href="{escape(item.html_href, quote=True)}">
  <div class="gallery-card__imgwrap">
    <img src="{escape(html_src(img), quote=True)}" alt="{escape(item.source_name, quote=True)}" loading="lazy">
    <div class="gallery-card__overlay">
      <div class="gallery-card__title">{escape(item.title)}</div>
    <div class="gallery-card__tags">{tag_chips(item.tags)}</div>
    </div>
  </div>
</a>
""".strip()
        )

    toc_text = ""
    if items:
        entries = "\n".join(f"   {item.rst_doc}" for item in items)
        toc_text = f"""
.. toctree::
   :maxdepth: 1
   :hidden:

{entries}
"""

    gallery_html = f"""
<section class="examples-section">
  <div class="gallery-grid">
    {''.join(cards)}
  </div>
</section>
""".strip()

    title = "Tutorials"
    underline = "=" * len(title)

    rst = f"""\
{title}
{underline}

{desc}

.. raw:: html

{indent(gallery_html, 3)}
{toc_text}
"""
    (TUTORIALS_DOCS / "index.rst").write_text(rst, encoding="utf-8")


def clean_previous_outputs() -> None:
    if not MANIFEST_FILE.exists():
        return

    try:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    for rel_path in manifest.get("generated", []):
        path = PROJECT_ROOT / rel_path
        if path.exists() and path.is_file():
            path.unlink()


def write_manifest(items: list[TutorialItem]) -> None:
    generated = [TUTORIALS_DOCS / "index.rst"]
    generated.extend(TUTORIALS_DOCS / f"{item.rst_doc}.rst" for item in items)
    generated.extend(STATIC_TUTORIALS / Path(item.source_web).relative_to("_static/tutorials") for item in items)
    generated.extend(
        STATIC_TUTORIALS / Path(item.img_web).relative_to("_static/tutorials")
        for item in items
        if item.img_web
    )

    rel_paths = sorted({path.relative_to(PROJECT_ROOT).as_posix() for path in generated})
    metadata_path = TUTORIALS_DOCS / ".content_items.json"
    metadata_path.write_text(
        json.dumps(
            [{"title": item.title, "href": item.html_href, "tags": item.tags, "type": "tutorial"} for item in items],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rel_paths.append(metadata_path.relative_to(PROJECT_ROOT).as_posix())
    MANIFEST_FILE.write_text(json.dumps({"generated": sorted(set(rel_paths))}, indent=2) + "\n", encoding="utf-8")


def prune_stale_pages(items: list[TutorialItem]) -> None:
    keep = {"index.rst"}
    keep.update(f"{item.rst_doc}.rst" for item in items)

    for rst_path in TUTORIALS_DOCS.glob("*.rst"):
        if rst_path.name not in keep:
            rst_path.unlink()


def main() -> None:
    ensure_dir(TUTORIALS_DOCS)
    ensure_dir(STATIC_TUTORIALS)
    clean_previous_outputs()

    items = collect_items()
    for item in items:
        write_tutorial_page(item)

    write_tutorials_landing(items)
    prune_stale_pages(items)
    write_manifest(items)

    print(f"Generated {len(items)} tutorial pages.")
    print(f"- Landing: {TUTORIALS_DOCS / 'index.rst'}")
    print(f"- Static files: {STATIC_TUTORIALS}/*")


if __name__ == "__main__":
    main()
