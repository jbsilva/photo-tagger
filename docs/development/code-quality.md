---
icon: lucide/shield-check
---

# Code quality

photo-tagger relies on a small set of linters, type checkers, and security scanners to keep the code
consistent and safe. Most of them run both locally (through pre-commit hooks) and in CI, so problems
surface before review.

## Tools

Each tool owns one slice of quality. Run them from the repository root.

### ruff

`ruff` handles both linting and formatting. The project sets a 100-column width and enables the full
rule set (`select = ALL`), so ruff is the first place style and lint issues show up.

Lint and auto-fix what it can, then format the tree:

```bash
uv run ruff check --fix .
uv run ruff format .
```

### zuban

`zuban` is the strict, mypy-compatible type checker. It catches type mismatches that ruff does not.

```bash
zuban check
```

### pycroscope

`pycroscope` is a semi-static analyzer that complements `zuban` with additional checks. Its config
lives under `[tool.pycroscope]` in `pyproject.toml`.

```bash
uv run pycroscope --config-file pyproject.toml
```

!!! warning

    `pycroscope` does not auto-discover the config file. Always pass `--config-file pyproject.toml` when
    invoking it directly, otherwise it runs with the wrong settings. The pre-commit hook already passes
    this flag for you.

### bandit

`bandit` scans the source for common security problems (for example, unsafe subprocess use or hard
coded secrets). It runs as a pre-commit hook and in CI.

### SonarCloud and SonarLint

SonarCloud analysis runs in CI as part of the test workflow. SonarLint surfaces the same rules in
your editor so you can catch findings before pushing. SonarQube has many false positives; see the
warning below before silencing anything.

## Pre-commit hooks

The repository uses `prek` to run pre-commit hooks. Run every hook against all files with one
command:

```bash
prek run -a
```

The configured hooks are:

| Hook                       | What it checks                                                |
| -------------------------- | ------------------------------------------------------------- |
| pre-commit-hooks basics    | Whitespace, end-of-file, merge conflicts, and similar basics. |
| uv-lock                    | Keeps `uv.lock` in sync with `pyproject.toml`.                |
| ruff-check                 | Lints with ruff and applies auto-fixes.                       |
| ruff-format                | Formats code to the 100-column style.                         |
| mdformat                   | Formats Markdown (a base pass plus a docs pass for `docs/`).  |
| bandit                     | Scans for common security issues.                             |
| pyupgrade                  | Rewrites code to modern syntax (`--py314-plus`).              |
| cspell                     | Spell-checks code and docs.                                   |
| typos                      | Catches common typos.                                         |
| renovate-config-validator  | Validates the Renovate config.                                |
| zuban (local)              | Strict type check.                                            |
| pycroscope (local)         | Semi-static analysis with the project config.                 |
| check-version-sync (local) | Keeps the package version consistent across project files.    |

The `check-version-sync` hook checks the version in `pyproject.toml`, `uv.lock`, `SECURITY.md`, and
`CHANGELOG.md`. The package itself no longer hardcodes a version: `photo_tagger.__version__` reads
the installed distribution metadata, so `pyproject.toml` is the only code-side source.

## Before committing

Run the full local sequence before each commit. The pre-commit hooks cover most of it, but running
the tools directly gives clearer output when something fails:

```bash
uv run ruff check --fix .
uv run ruff format .
zuban check
uv run pycroscope --config-file pyproject.toml
prek run -a
```

!!! warning

    Never silence a lint, type, or SonarQube finding just to make CI pass. Fix the underlying cause.
    SonarQube does produce false positives, so silencing is allowed only when the finding clearly does
    not apply, and only with a one-line reason explaining why.

See [Testing](testing.md) for the test suite and coverage target, and
[Dependencies](dependencies.md) for how updates are managed.
