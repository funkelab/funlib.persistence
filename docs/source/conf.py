project = "funlib.persistence"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
    "myst_parser",
]
# Styled after the e11lib.ml docs: pydata theme, no sidebars, full width, and
# the system's light or dark mode.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 3,
    "navigation_depth": 3,
    "collapse_navigation": True,
}
html_sidebars = {"**": []}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
autodoc_member_order = "bysource"
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
myst_heading_anchors = 3
