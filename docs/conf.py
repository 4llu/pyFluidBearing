# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'pyFluidBearing'
copyright = '2025, Aleksanteri Hämäläinen, Oliver Häggman, Huy Nguyen, Elmo Laine'
author = 'Aleksanteri Hämäläinen, Oliver Häggman, Huy Nguyen, Elmo Laine'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_baseurl = "https://4llu.github.io/pyFluidBearing/"

# Use relative URLs for all static files (critical for GitHub Pages)
html_static_path = ['_static']
html_css_files = []
html_js_files = []

html_copy_source = True
html_show_sourcelink = True
html_add_permalinks = True

html_use_index = True
html_domain_indices = True