# Configuration file for the Sphinx documentation builder.

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

extensions = []

templates_path = ['_templates']
exclude_patterns = []

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
    ".rst": "restructuredtext"
    #".md": "markdown",
}

html_theme_options = {
    "logo": {"text": "baltic"},
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

html_context = {}

html_additional_pages = {
    "index": "index.html",
}
