---
icon: lucide/shield-check
---

# Code quality

photo-tagger uses a small set of linters, type checkers, and security scanners to keep the code
consistent and safe. Most of them run both locally (through pre-commit hooks) and in CI, so problems
appear before review.

## Tools

Each tool covers one area of quality. Run the tools from the repository root.

### ruff

`ruff` handles both linting and formatting. The project sets a 100-column width and enables the full
rule set (`select = ALL`). As a result, style and lint issues appear first in ruff.

Lint the code and auto-fix what ruff can. Then format the tree:

```bash
uv run ruff check --fix .
uv run ruff format .
```

### zuban

`zuban` is the strict, mypy-compatible type checker. It catches type mismatches that ruff does not
catch.

```bash
uv run zuban check
```

!!! note

    Use `uv run`, not a bare `zuban check`. A global `uv tool install zuban` on `PATH` can shadow the
    version pinned in `uv.lock`.

### pycroscope

`pycroscope` is a semi-static analyzer that complements `zuban` with more checks. Its configuration
lives under `[tool.pycroscope]` in `pyproject.toml`.

```bash
uv run pycroscope --config-file pyproject.toml
```

!!! warning

    If you run `pycroscope` directly, always pass `--config-file pyproject.toml`. The tool does not
    auto-discover the configuration file. Without this flag, it runs with the wrong configuration. The
    pre-commit hook already passes this flag for you.

!!! note "The GUI package is a static-analysis exception"

    `src/photo_tagger/gui/` is the one package excluded from the type checker, the analyzer, and
    coverage. PySide6 builds its widget attributes at runtime through `shiboken`. As a result, the
    static tools report false `undefined_attribute` errors on every `.addWidget`, `.clicked`, and the
    like. For this reason, each module in the package starts with two headers. The
    `# mypy: ignore-errors` header excludes the module from `zuban`, and the `# static analysis: ignore`
    header excludes it from `pycroscope`. `[tool.coverage.run].omit` excludes the whole package from
    coverage.

    The testable, Qt-free logic lives in `gui_state.py`. The type checker, the analyzer, and coverage
    treat that file normally. This is the documented exception to the "fix, do not silence" rule below.
    You cannot model a Qt event-loop shell statically or run it headlessly. See
    [Testing](testing.md#gui-tests) for how the project tests it instead.

### bandit

`bandit` scans the source for common security problems (for example, unsafe subprocess use or hard
coded secrets). It runs as a pre-commit hook and in CI.

### zizmor

[`zizmor`](https://docs.zizmor.sh) audits the GitHub Actions workflows themselves. No Python linter
looks at these files. It catches the mistakes that are specific to CI:

- Credentials left behind in the checkout
- Caches that a lower-privileged run can poison
- Untrusted input interpolated into a `run:` block
- Jobs that hold more permissions than they use

```bash
uv run zizmor .
```

It runs as a pre-commit hook on anything under `.github/workflows/`. It also runs in the
`zizmor.yml` workflow, which sends its findings to the Security tab of the repository.

CI passes a `GH_TOKEN` to switch on the audits that query the GitHub API. These audits report a
referenced action that is vulnerable or whose tag moved. Without a token, those audits skip quietly,
and the local run does only offline checks.

### SonarCloud and SonarLint

SonarCloud analysis runs in CI as part of the test workflow. SonarLint shows the same rules in your
editor, so you can catch findings before you push. SonarQube has many false positives. Read the
warning below before you silence anything.

## Pre-commit hooks

The repository uses [`prek`](https://prek.j178.dev) to run the git hooks. `prek.toml` configures the
hooks in the native TOML format of prek. The repository does not use `.pre-commit-config.yaml`, the
file that the original `pre-commit` tool reads. Run every hook against all files with one command:

```bash
prek run -a
```

The configured hooks are:

| Hook                       | What it checks                                                |
| -------------------------- | ------------------------------------------------------------- |
| pre-commit-hooks basics    | Whitespace, end-of-file, merge conflicts, and similar basics. |
| uv-lock                    | Keeps `uv.lock` in sync with `pyproject.toml`.                |
| docformatter               | Reflows docstrings to 100 columns (runs before ruff).         |
| ruff-check                 | Lints with ruff and applies auto-fixes.                       |
| ruff-format                | Formats code to the 100-column style.                         |
| mdformat                   | Formats Markdown (a base pass plus a docs pass for `docs/`).  |
| bandit                     | Scans for common security issues.                             |
| pyupgrade                  | Rewrites code to modern syntax (`--py314-plus`).              |
| cspell                     | Spell-checks code and docs.                                   |
| typos                      | Catches common typos.                                         |
| renovate-config-validator  | Makes sure that the Renovate configuration is valid.          |
| zizmor                     | Audits the GitHub Actions workflows for security.             |
| zuban (local)              | Strict type check.                                            |
| pycroscope (local)         | Semi-static analysis with the project configuration.          |
| check-version-sync (local) | Keeps the package version consistent across project files.    |
| check-docs-sync (local)    | Keeps the documented CLI surface matching the real app.       |
| check-translations (local) | Fails on a `.po` change that has no recompiled catalog.       |

The local hooks all run through `uv run`. As a result, they use the versions pinned in `uv.lock`,
not whatever version is first on `PATH`.

The `check-version-sync` hook makes sure that the version is the same in `pyproject.toml`,
`uv.lock`, `SECURITY.md`, and `CHANGELOG.md`. The package itself no longer hardcodes a version.
`photo_tagger.__version__` reads the installed distribution metadata, so `pyproject.toml` is the
only code-side source.

## Before committing

Run the full local sequence before each commit. The pre-commit hooks cover most of it. If something
fails, a direct run of the tools gives clearer output:

```bash
uv run poe check
```

The `check` task runs these tasks in order and stops at the first failure:

```bash
uv run poe lint         # ruff check --fix .
uv run poe format       # ruff format .
uv run poe typecheck    # zuban check
uv run poe pycroscope   # pycroscope --config-file pyproject.toml
uv run poe hooks        # prek run -a
```

!!! warning

    Never silence a lint, type, or SonarQube finding only to make CI pass. Fix the underlying cause.
    SonarQube does produce false positives. If a finding clearly does not apply, you can silence it, but
    only with a one-line reason that explains why.

See [Testing](testing.md) for the test suite and coverage target, and
[Dependencies](dependencies.md) for how the project manages updates.
