#!/usr/bin/env python3
from __future__ import annotations

import re
import json
from html import escape
import shutil
from dataclasses import dataclass
from pathlib import Path

from content_metadata import load_metadata, tag_chips, tags_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXAMPLES_SRC = PROJECT_ROOT / "source" / "baltic-examples"          # input
SPHINX_SOURCE = PROJECT_ROOT / "source"
EXAMPLES_DOCS = SPHINX_SOURCE / "examples"                           # output rst
STATIC_EXAMPLES = SPHINX_SOURCE / "_static" / "examples"             # output images
DESC_FILE = SPHINX_SOURCE / "_data" / "examples_description.txt"     # manual text

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SOURCE_EXTS = {".py", ".ipynb"}
MANIFEST_FILE = EXAMPLES_DOCS / ".build_examples_manifest.json"


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


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


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


def read_source_code(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return notebook_code(path)
    return safe_read_text(path)


def title_from_stem(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ")


def find_matching_image(source: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = source.with_suffix(ext)
        if candidate.exists():
            return candidate

    source_stem = source.stem.lower()
    images = [
        p for p in source.parent.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem.lower() == source_stem
    ]
    return sorted(images, key=lambda p: p.name.lower())[0] if images else None


def copy_static_file(source: Path, relative_parent: Path) -> str:
    parent_slug = "-".join(slugify(part) for part in relative_parent.parts)
    if not parent_slug:
        parent_slug = "root"

    out_dir = STATIC_EXAMPLES / parent_slug
    ensure_dir(out_dir)

    dest = out_dir / source.name
    shutil.copy2(source, dest)
    return f"_static/examples/{parent_slug}/{source.name}"


def unique_doc_slug(base_slug: str, used: set[str]) -> str:
    candidate = base_slug
    counter = 2
    while candidate in used:
        candidate = f"{base_slug}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


@dataclass
class ExampleItem:
    category_name: str
    category_slug: str
    relative_parent: Path
    example_name: str
    example_slug: str
    img_web: str | None          # _static/examples/<cat>/<file>.png
    source_web: str              # _static/examples/<cat>/<file>.py or .ipynb
    source_suffix: str
    rst_doc: str                 # <cat>/<example>
    html_href: str               # <cat>/<example>.html
    source_path: Path
    tags: list[str]


def collect_items() -> tuple[list[ExampleItem], dict[str, list[ExampleItem]]]:
    items: list[ExampleItem] = []
    by_cat: dict[str, list[ExampleItem]] = {}
    used_doc_slugs: set[str] = set()
    metadata = load_metadata("examples")

    if not EXAMPLES_SRC.exists():
        raise SystemExit(f"Missing input folder: {EXAMPLES_SRC}")

    sources = [
        p for p in EXAMPLES_SRC.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SOURCE_EXTS
        and not is_hidden_path(p.relative_to(EXAMPLES_SRC))
    ]
    sources.sort(key=lambda p: p.relative_to(EXAMPLES_SRC).as_posix().lower())

    for source in sources:
        rel = source.relative_to(EXAMPLES_SRC)
        relative_parent = rel.parent
        category_name = rel.parts[0] if len(rel.parts) > 1 else "Examples"
        category_slug = slugify(category_name)
        example_name = source.stem
        example_slug = slugify(example_name)

        slug_parts = [slugify(part) for part in rel.with_suffix("").parts]
        doc_slug = unique_doc_slug("-".join(slug_parts), used_doc_slugs)

        img = find_matching_image(source)
        img_web = copy_static_file(img, relative_parent) if img else None
        source_web = copy_static_file(source, relative_parent)

        it = ExampleItem(
            category_name=category_name,
            category_slug=category_slug,
            relative_parent=relative_parent,
            example_name=example_name,
            example_slug=example_slug,
            img_web=img_web,
            source_web=source_web,
            source_suffix=source.suffix.lower(),
            rst_doc=doc_slug,
            html_href=f"{doc_slug}.html",
            source_path=source,
            tags=tags_for(metadata, rel.as_posix(), category_name),
        )
        items.append(it)
        by_cat.setdefault(category_slug, []).append(it)

    # stable ordering
    for k in by_cat:
        by_cat[k].sort(key=lambda x: x.source_path.relative_to(EXAMPLES_SRC).as_posix().lower())

    return items, by_cat


def write_example_page(item: ExampleItem) -> None:
    ensure_dir(EXAMPLES_DOCS)

    code = read_source_code(item.source_path)

    hero = item.img_web or "_static/no_image.png"
    source_kind = "Notebook" if item.source_suffix == ".ipynb" else "Python source"
    download_label = escape(f"Download {source_kind.lower()}")

    title = title_from_stem(item.example_name)
    underline = "=" * len(title)
    indented_code = indent(code.rstrip() + "\n", 3)

    rst = f"""\
{title}
{underline}

.. image:: {rst_static_path(hero)}
   :alt: {item.example_name}
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
    (EXAMPLES_DOCS / f"{item.rst_doc}.rst").write_text(rst, encoding="utf-8")


def write_examples_landing(by_cat: dict[str, list[ExampleItem]], items: list[ExampleItem]) -> None:
    ensure_dir(EXAMPLES_DOCS)

    desc = ""
    if DESC_FILE.exists():
        desc = read_text(DESC_FILE)

    # Build raw HTML sections per category
    sections_html: list[str] = []
    for cat_slug in sorted(by_cat.keys()):
        cat_items = by_cat[cat_slug]
        cat_name = cat_items[0].category_name if cat_items else cat_slug

        cards: list[str] = []
        for it in cat_items:
            img = it.img_web or "_static/no_image.png"
            title = escape(title_from_stem(it.example_name))
            alt = escape(it.example_name, quote=True)
            href = escape(it.html_href, quote=True)
            src = escape(html_src(img), quote=True)
            cards.append(
                f"""
<div class="gallery-card">
  <div class="gallery-card__imgwrap">
        <a class="gallery-card__image-link" href="{href}">
            <img src="{src}" alt="{alt}" loading="lazy">
        </a>
    <div class="gallery-card__overlay">
            <a class="gallery-card__title" href="{href}">{title}</a>
            <div class="gallery-card__tags">{tag_chips(it.tags)}</div>
    </div>
  </div>
</div>
""".strip()
            )

        section = f"""
<section class="examples-section">
  <h2 class="examples-section__title">{escape(cat_name)}</h2>
  <div class="gallery-grid">
    {''.join(cards)}
  </div>
</section>
""".strip()
        sections_html.append(section)

    # Keep one hidden toctree for the landing page sidebar, as in tutorials.
    # Category names remain visible in the gallery while every example stays
    # directly available from the section navigation.
    toc_entries: list[str] = []
    for cat_slug in sorted(by_cat.keys()):
        cat_items = by_cat[cat_slug]
        if not cat_items:
            continue

        toc_entries.extend(f"   {it.rst_doc}" for it in cat_items)

    toc_text = ""
    if toc_entries:
        toc_text = f"""
.. toctree::
   :maxdepth: 1
   :hidden:

{chr(10).join(toc_entries)}
"""

    title = "Examples"
    underline = "=" * len(title)

    rst = f"""\
{title}
{underline}

{desc}

.. raw:: html

{indent("\n".join(sections_html), 3)}

{toc_text}
"""
    (EXAMPLES_DOCS / "index.rst").write_text(rst, encoding="utf-8")


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


def write_manifest(items: list[ExampleItem]) -> None:
    generated = [EXAMPLES_DOCS / "index.rst"]
    generated.extend(EXAMPLES_DOCS / f"{item.rst_doc}.rst" for item in items)
    generated.extend(STATIC_EXAMPLES / Path(item.source_web).relative_to("_static/examples") for item in items)
    generated.extend(
        STATIC_EXAMPLES / Path(item.img_web).relative_to("_static/examples")
        for item in items
        if item.img_web
    )

    rel_paths = sorted({path.relative_to(PROJECT_ROOT).as_posix() for path in generated})
    metadata_path = EXAMPLES_DOCS / ".content_items.json"
    metadata_path.write_text(
        json.dumps(
            [{"title": title_from_stem(item.example_name), "href": item.html_href, "image": item.img_web, "tags": item.tags, "type": "example"} for item in items],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rel_paths.append(metadata_path.relative_to(PROJECT_ROOT).as_posix())
    MANIFEST_FILE.write_text(json.dumps({"generated": sorted(set(rel_paths))}, indent=2) + "\n", encoding="utf-8")


def prune_stale_pages(items: list[ExampleItem]) -> None:
    keep = {"index.rst"}
    keep.update(f"{item.rst_doc}.rst" for item in items)

    for rst_path in EXAMPLES_DOCS.glob("*.rst"):
        if rst_path.name not in keep:
            rst_path.unlink()


def main() -> None:
    ensure_dir(EXAMPLES_DOCS)
    ensure_dir(STATIC_EXAMPLES)
    clean_previous_outputs()

    items, by_cat = collect_items()

    # write pages
    for it in items:
        write_example_page(it)

    # write landing page
    write_examples_landing(by_cat, items)
    prune_stale_pages(items)
    write_manifest(items)

    print(f"Generated {len(by_cat)} categories, {len(items)} example pages.")
    print(f"- Landing: {EXAMPLES_DOCS / 'index.rst'}")
    print(f"- Static files: {STATIC_EXAMPLES}/<source-folder>/*")


if __name__ == "__main__":
    main()
