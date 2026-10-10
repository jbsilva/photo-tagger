---
icon: lucide/heart-handshake
---

# Contributing

Contributions are welcome: a bug report, a documentation fix, a new recipe, or a code change. This
page covers the conventions that keep the codebase consistent. It also covers the checks that every
pull request must pass.

## Set up a dev environment

Before you write any code, set up a working environment. Install the dependencies and the git hooks
in it. The [Development](development/index.md) section gives the steps for both the dev container
and the local setup. It also lists the requirements: Python 3.14+, ExifTool, and libraw for RAW
decoding.

## Code style

The code follows a few conventions that the linters and reviewers expect:

- Python 3.14+ modern syntax: PEP 695 generics, `match` and structural pattern matching, and
    `except` without parentheses are all permitted.
- 100-column width for code, comments, and docstrings.
- Plain, simple language in comments and docstrings. Explain the *why*, not the *what*.
- Small, focused functions: if a function grows deeply nested or hard to test, refactor it before
    you add more to it. Extract helpers and split responsibilities.
- Never use an em-dash or an en-dash. Rephrase with a colon, a semicolon, parentheses, or two
    sentences. A spaced hyphen is not a substitute.

[`pyproject.toml`](https://github.com/jbsilva/photo-tagger/blob/main/pyproject.toml) holds the ruff
configuration that enforces formatting and most of these rules.

## Tests

Each time that you change behavior, add or update tests. The suite targets 90%+ coverage. As a
result, a new code path usually needs a new test of its own.

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

The hooks do many checks. These are the ones that you will hit most often:

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

    Never skip the hooks with `--no-verify`. Never silence a lint, type, or SonarQube finding only to
    make CI pass. If a hook fails, fix the underlying cause. If a finding clearly does not apply, you
    can silence it. Then document the reason.

See [Code quality](development/code-quality.md) for the details of each tool and how to run them on
their own.

## Commits

Make small, atomic commits: one logical change per commit. Commit as you go. Do not put many
unrelated edits into one commit at the end. Write a short imperative subject line. In the body,
explain *why* the change is necessary, not only what changed.

## Pull requests and CI

Three GitHub Actions workflows gate every pull request:

- [`tests.yml`](https://github.com/jbsilva/photo-tagger/blob/main/.github/workflows/tests.yml): runs
    the unit tests, the linters, and the SonarCloud analysis.
- [`codeql.yml`](https://github.com/jbsilva/photo-tagger/blob/main/.github/workflows/codeql.yml):
    runs CodeQL security scanning.
- [`zizmor.yml`](https://github.com/jbsilva/photo-tagger/blob/main/.github/workflows/zizmor.yml):
    audits the workflow files themselves. A change to any file under `.github/workflows/` starts it.

Run the local checks before you push, so that CI finds nothing that you can catch locally.

!!! tip

    Before you commit, run the same checks that CI runs:

    ```bash
    uv run poe check    # ruff, zuban, pycroscope, import-linter, then all pre-commit hooks
    ```

Never push secrets. Keep API keys and other credentials out of commits and pull requests. Pass them
through environment variables instead, for example `OLLAMA_API_KEY` or `LM_STUDIO_API_KEY`.
