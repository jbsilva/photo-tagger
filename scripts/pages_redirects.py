#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""
Write a redirect page for each page of the built docs, for the old GitHub Pages address.

The docs moved from https://jbsilva.github.io/photo-tagger/ to https://docs.tagger.photo/. Links
to the old address live on in released packages (the PyPI page, the telemetry notice), and GitHub
Pages cannot send a real redirect. So each old page becomes a small HTML page that sends the reader
to the same page at the new address, and 404.html does the same for a page that no longer exists.

Usage: pages_redirects.py SITE_DIR OUT_DIR
"""

import html
import sys
from pathlib import Path


NEW_BASE = "https://docs.tagger.photo/"
# The path of the project site on GitHub Pages, which the 404 page strips from the old URL.
OLD_PREFIX = "/photo-tagger/"

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moved to {shown}</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace({script_target} + location.search + location.hash);</script>
</head>
<body><p>The documentation moved to <a href="{target}">{shown}</a>.</p></body>
</html>
"""

# GitHub Pages serves this for any path it does not have, so the target comes from the URL itself.
_NOT_FOUND_SCRIPT = (
    f'"{NEW_BASE}" + location.pathname.replace(/^{OLD_PREFIX.replace("/", "\\/")}/, "")'
)


def new_url(page: Path) -> str:
    """Map a built page (relative to the site root) to its address on the new domain."""
    if page.name == "index.html":
        folder = page.parent.as_posix()
        return NEW_BASE if folder == "." else f"{NEW_BASE}{folder}/"
    return NEW_BASE + page.as_posix()


def redirect_page(target: str, script_target: str | None = None) -> str:
    """Render a page that sends the reader on to *target*, keeping the query and the anchor."""
    quoted = html.escape(target, quote=True)
    return _PAGE.format(
        shown=quoted,
        target=quoted,
        script_target=script_target or f'"{target}"',
    )


def write_redirects(site: Path, out: Path) -> int:
    """Write one redirect page per HTML page of *site* into *out*, and return how many."""
    count = 0
    for source in sorted(site.rglob("*.html")):
        page = source.relative_to(site)
        dest = out / page
        dest.parent.mkdir(parents=True, exist_ok=True)
        if page.as_posix() == "404.html":
            dest.write_text(redirect_page(NEW_BASE, _NOT_FOUND_SCRIPT))
        else:
            dest.write_text(redirect_page(new_url(page)))
        count += 1
    return count


def main() -> int:
    """Run the script from the command line."""
    if len(sys.argv) != 3:  # noqa: PLR2004 - the script name and two paths.
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)  # noqa: T201 - CLI usage.
        return 2
    count = write_redirects(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Wrote {count} redirect pages")  # noqa: T201 - CLI output.
    return 0


if __name__ == "__main__":
    sys.exit(main())
