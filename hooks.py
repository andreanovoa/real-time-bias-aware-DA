"""MkDocs build hooks for the tutorial notebooks.

Converts notebooks to markdown at build time via nbconvert, writing directly
into docs/tutorials/<folder>/ (gitignored, generated) so they render through
the site's normal Python-Markdown + pymdownx.arithmatex pipeline, same as
every other page -- math included. mkdocs-jupyter's own HTML export bypasses
that pipeline entirely (its output isn't wrapped in the `arithmatex` class our
MathJax config restricts itself to), which is why LaTeX wasn't rendering
there. This also avoids staging a byte-for-byte copy of each source .ipynb
inside docs_dir, which the mkdocs-jupyter plugin required.

Raw HTML <img>/<video> tags authored in the notebooks (pointing at
../../../docs/figs/...) pass through nbconvert unrewritten; a post-build pass
over the built HTML recomputes each page's actual depth and fixes them, since
MkDocs' own relative-link resolution only rewrites markdown-authored links,
not arbitrary raw HTML content.
"""
import re
import shutil
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter

REPO_ROOT = Path(__file__).parent
SRC = REPO_ROOT / "scripts" / "tutorials"
DEST = REPO_ROOT / "docs" / "tutorials"

# Natively rendered for now; 2_Real-time_DA_Thermoacoustics, 3_Introduction_to_ROMs
# and 4_Real-time_DA_ROMs stay as GitHub links in tutorials.md/nav until we're
# ready to pay their build-time and site-size cost.
RENDERED_TOPICS = ["0_How_to_repo", "1_Introduction_to_real-time_DA"]

# Matches a (possibly already-mangled) relative reference to the figs/ dir,
# with or without a literal "docs/" segment, so it self-corrects regardless
# of how many ../ segments survived the conversion.
FIGS_HREF_RE = re.compile(r'(src|href)="(?:\.\./)*(?:docs/)?figs/([^"]*)"')

# Bare, same-directory cross-notebook links, e.g. ](00_Class_Model.ipynb).
NOTEBOOK_LINK_RE = re.compile(r"\]\(([\w-]+)\.ipynb\)")


def on_pre_build(config, **kwargs):
    exporter = MarkdownExporter()
    rendered_stems = {p.stem for t in RENDERED_TOPICS for p in (SRC / t).glob("*.ipynb")}

    def fix_notebook_link(m):
        stem = m.group(1)
        # Within the rendered set: point at the sibling .md page. Otherwise
        # (topics 2/3, not rendered yet): fall back to viewing it on GitHub.
        if stem in rendered_stems:
            return f"]({stem}.md)"
        nb_path = next(SRC.rglob(f"{stem}.ipynb"), None)
        folder = nb_path.parent.name if nb_path else ""
        return f"](https://github.com/andreanovoa/real-time-bias-aware-DA/blob/main/scripts/tutorials/{folder}/{stem}.ipynb)"

    for topic in RENDERED_TOPICS:
        topic_dest = DEST / topic
        if topic_dest.exists():
            shutil.rmtree(topic_dest)
        topic_dest.mkdir(parents=True)

        for nb_path in sorted((SRC / topic).glob("*.ipynb")):
            nb = nbformat.read(nb_path, as_version=4)
            body, resources = exporter.from_notebook_node(
                nb, resources={"output_files_dir": f"{nb_path.stem}_files"}
            )
            body = NOTEBOOK_LINK_RE.sub(fix_notebook_link, body)
            FilesWriter(build_directory=str(topic_dest)).write(
                body, resources, notebook_name=nb_path.stem
            )


def on_post_build(config, **kwargs):
    site_dir = Path(config["site_dir"])
    for html_path in site_dir.rglob("*.html"):
        depth = len(html_path.relative_to(site_dir).parent.parts)
        up = "../" * depth

        text = html_path.read_text(encoding="utf-8")
        fixed = FIGS_HREF_RE.sub(lambda m: f'{m.group(1)}="{up}figs/{m.group(2)}"', text)
        if fixed != text:
            html_path.write_text(fixed, encoding="utf-8")
