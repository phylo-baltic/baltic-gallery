# Configuration file for the Sphinx documentation builder.

from datetime import date
from pathlib import Path
import re

from sphinx.errors import ConfigError

# -- Image import path

#import os
#html_extra_path = [
#    name for name in os.listdir(".")
#    if os.path.isdir(name) and name not in ["_static", "_templates"]
#]
html_extra_path = []

# -- Project information -----------------------------------------------------

project = 'baltic'
author = 'mft'
copyright = '2025'
release = '1'

# -- General configuration ---------------------------------------------------

extensions = ['myst_parser']

templates_path = ['_templates']
exclude_patterns = [
    'baltic-examples/*.md',
    'baltic-examples/**/*.md',
]

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# --- CSS: Only custom styling you actually need ----------------------------
html_css_files = [
    'css/normalize.css',
    # 'css/landing.css',
    'css/style.css',
    'css/examples-grid.css',
    'vendor/fontawesome/6.5.2/css/all.min.css',  # FontAwesome icons
]

# --- JS: ONLY custom homepage JS, NOTHING from Sphinx/theme -----------------
html_js_files = [
    'rotator-images.js',
    'js/examples-grid.js',
    'script.js',
]

# Theme options
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme_options = {
    "logo": {"text": "baltic"},
    "navbar_persistent": [],  # Remove the search bar on desktop and mobile.
    "secondary_sidebar_items": {
        "**": [],
    },
    "show_nav_level": 0,        # only show captions
    "collapse_navigation": True,
    "navigation_depth": 1,      # captions -> pages (your grandchildren) are depth 1
}

html_sidebars = {
    "index": [],   # no primary (left) sidebar on homepage
    "uses/index": [],  # no primary sidebar on the "As seen in" page
}

SOURCE_DIR = Path(__file__).resolve().parent
NEWS_DIR = SOURCE_DIR / "news"
NEWS_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")
NEWS_TITLE_PATTERN = re.compile(r"^#\s+(.+?)\s*$")
NEWS_MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def load_news_entries():
    """Read news metadata from Markdown files and return newest entries first."""
    entries = []

    for path in sorted(NEWS_DIR.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].strip() != "---":
            raise ConfigError(f"{path}: news articles must start with YAML front matter")

        try:
            front_matter_end = next(
                index for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            )
        except StopIteration as error:
            raise ConfigError(f"{path}: YAML front matter is missing its closing ---") from error

        date_values = [
            line.split(":", 1)[1].strip().strip("\"'")
            for line in lines[1:front_matter_end]
            if line.split(":", 1)[0].strip() == "date" and ":" in line
        ]
        if len(date_values) != 1 or not NEWS_DATE_PATTERN.fullmatch(date_values[0]):
            raise ConfigError(
                f"{path}: front matter must contain one date in YYYY-MM or YYYY-MM-DD format"
            )

        date_iso = date_values[0]
        try:
            published = date.fromisoformat(
                date_iso if len(date_iso) == 10 else f"{date_iso}-01"
            )
        except ValueError as error:
            raise ConfigError(f"{path}: {date_iso!r} is not a valid date") from error

        filename_prefix = (
            f"{published.year}-{NEWS_MONTH_NAMES[published.month - 1]}-"
        )
        if not path.name.startswith(filename_prefix):
            raise ConfigError(
                f"{path}: filename must start with the article year and month "
                f"({filename_prefix})"
            )

        title = next(
            (
                match.group(1)
                for line in lines[front_matter_end + 1:]
                if (match := NEWS_TITLE_PATTERN.fullmatch(line.strip()))
            ),
            None,
        )
        if title is None:
            raise ConfigError(f"{path}: news articles must contain a level-one Markdown heading")

        entries.append({
            "title": title,
            "date_iso": date_iso,
            "date_display": published.strftime(
                "%d %b %Y" if len(date_iso) == 10 else "%b %Y"
            ),
            "docname": path.relative_to(SOURCE_DIR).with_suffix("").as_posix(),
            "published": published,
        })

    entries.sort(key=lambda entry: entry["title"].casefold())
    entries.sort(key=lambda entry: entry["published"], reverse=True)
    return entries


html_context = {
    "news_entries": load_news_entries(),
}

html_additional_pages = {
    "index": "index.html",
}
