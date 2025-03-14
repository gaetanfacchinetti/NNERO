# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'NNERO'
copyright = '2024, Gaétan Facchinetti'
author = 'Gaétan Facchinetti'
release = '1.0.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", 
              "sphinx_markdown_builder", 
              "sphinx.ext.todo", 
              "sphinx.ext.viewcode", 
              "sphinx.ext.autodoc", 
              "sphinx.ext.napoleon", 
              "sphinx.ext.autosummary",
              "sphinx.ext.githubpages"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def skip_properties(app, what, name, obj, skip, options):
    """
    Skip members decorated with @property.
    """
    if isinstance(obj, property):
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_properties)
