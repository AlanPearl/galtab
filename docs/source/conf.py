# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from pkg_resources import DistributionNotFound, get_distribution

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
try:
    __version__ = get_distribution("galtab").version
except:
    __version__ = "unknown version"

project = 'galtab'
copyright = '2023, Alan Pearl'
author = 'Alan Pearl'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = [".ipynb_checkpoints/*"]

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'
html_static_path = ['_static']
