---
icon: lucide/heart-handshake
---

# Contributing

Contributions are welcome, whether that is a bug report, a documentation fix, a new recipe, or a
code change. This page covers the conventions that keep the codebase consistent and the checks that
gate every pull request.

## Set up a dev environment

Before writing any code, get a working environment with the dependencies and git hooks installed.
The [Development](development/index.md) section walks through both the dev container and the local
setup, and lists the requirements (Python 3.14+, ExifTool, and libraw for RAW decoding).

## Code style

The code follows a few conventions that the linters and reviewers expect:

- **Python 3.14+ modern syntax.** PEP 695 generics, `match` and structural pattern matching, and
    `except` without parentheses are all fair game.
- **100-column width** for code, comments, and docstrings.
- **Plain, simple language** in comments and docstrings. Explain the *why*, not the *what*.
- **Small, focused functions.** If a function grows deeply nested or hard to test, refactor before
    adding more to it: extract helpers and split responsibilities.
- **Never use an em-dash.** Use a regular hyphen or rephrase the sentence.

The ruff configuration that enforces formatting and most of these rules lives in
[`pyproject.toml`](https://github.com/jbsilva/photo-tagger/blob/main/pyproject.toml).

## Tests

Add or update tests whenever you change behavior. The suite targets 90%+ coverage, so a new code
path usually needs a new test to go with it.

Run the suite with the `test` task:

```bash
uv run poe test                 # Full suite
uv run poe test -k test_name    # A single test
```

See [Testing](development/testing.md) for markers, coverage, and how to write new tests.

## Pre-commit hooks

The git hooks catch most issues before they reach CI. Run every hook across the whole tree with:

```bash
uv run poe hooks
```

The hooks cover a range of checks; these are the ones you will hit most often:

| Hook                 | What it does                                          |
| -------------------- | ----------------------------------------------------- |
| `ruff-check`         | Lints the code (selects the full rule set).           |
| `ruff-format`        | Formats the code at 100 columns.                      |
| `zuban`              | Strict, mypy-compatible type checking.                |
| `pycroscope`         | Semi-static analysis that complements zuban.          |
| `bandit`             | Flags common security issues.                         |
| `zizmor`             | Audits the GitHub Actions workflows for security.     |
| `uv-lock`            | Keeps `uv.lock` in sync with `pyproject.toml`.        |
| `check-version-sync` | Keeps the docs version in sync with `pyproject.toml`. |

!!! warning

    Never skip the hooks with `--no-verify`, and never silence a lint, type, or SonarQube finding just
    to make CI pass. If a hook fails, fix the underlying cause. Silence a finding only when it clearly
    does not apply, and document the reason.

See [Code quality](development/code-quality.md) for the details of each tool and how to run them on
their own.

## Commits

Make small, atomic commits: one logical change each. Commit as you go rather than batching many
unrelated edits at the end. Write a short imperative subject line, and use the body to explain *why*
the change is needed, not just what changed.

## Pull requests and CI

Two GitHub Actions workflows gate every pull request:

- [`tests.yml`](https://github.com/jbsilva/photo-tagger/blob/main/.github/workflows/tests.yml): runs
    the unit tests, the linters, and the SonarCloud analysis.
- [`codeql.yml`](https://github.com/jbsilva/photo-tagger/blob/main/.github/workflows/codeql.yml):
    runs CodeQL security scanning.
- [`zizmor.yml`](https://github.com/jbsilva/photo-tagger/blob/main/.github/workflows/zizmor.yml):
    audits the workflow files themselves. Editing anything under `.github/workflows/` will run it.

Run the local checks before you push so CI does not surface anything you could have caught locally.

!!! tip

    Before you commit, run the same checks that CI runs:

    ```bash
    uv run poe check    # ruff, zuban, pycroscope, then all pre-commit hooks
    ```

Never push secrets. Keep API keys and other credentials out of commits and pull requests; pass them
through environment variables such as `OLLAMA_API_KEY` or `LM_STUDIO_API_KEY` instead.
