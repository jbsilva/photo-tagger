---
icon: lucide/package
---

# Dependencies

[uv](https://docs.astral.sh/uv/) manages the virtual environment of the project. The project commits
`uv.lock`, so every contributor and every CI run builds against exactly the same pinned set of
packages.

## uv and the lockfile

`uv.lock` records the resolved version of every direct and transitive dependency. Because the file
is in the repository, `uv sync` reproduces an identical environment everywhere. Renovate edits the
lock file together with `pyproject.toml`.

To keep resolution reproducible over time, `pyproject.toml` limits how new an artifact can be:

```toml
[tool.uv]
exclude-newer = "1 day"
```

`exclude-newer = "1 day"` tells uv to resolve only against packages published at least a day ago.
`uv lock` does not see a brand-new release until the release is older than that window. As a result,
uv does not lock onto an artifact that was just published. The publisher can still yank or repair a
release that new.

## Optional extras and dependency groups

The project splits its non-core dependencies in two ways. Each way has a different purpose:

- `[project.optional-dependencies]` are user-facing extras that ship with the package. There are
    two. `gui` (PySide6) is the desktop GUI. Users install it on demand with
    `pip install 'photo-tagger[gui]'` or `uv tool install 'photo-tagger[gui]'`. Because the GUI is
    an extra, the base install stays Qt-free. `otel` (the OpenTelemetry SDK and its OTLP/HTTP
    exporter) is what `--otel` needs.
- `[dependency-groups]` (`dev`, `test`, `docs`, `package`) are contributor tooling. `package` holds
    PyInstaller, for the macOS app bundle. The project does not publish them to end users. You sync
    them with `uv sync --group dev --group test`. CI installs the linters, type checker, test
    runner, and docs builder the same way.

To work on the GUI locally, sync the extra alongside the dev groups:

```bash
uv sync --extra gui --group dev --group test
```

## Renovate

[`renovate.json5`](https://github.com/jbsilva/photo-tagger/blob/main/renovate.json5) controls the
automated dependency updates. It replaces the old `.github/dependabot.yml`. It extends
`config:best-practices` and covers five update domains:

- `pyproject.toml` plus `uv.lock` (the Python dependencies).
- GitHub Actions in `.github/workflows/`, pinned to a full commit SHA with the version kept in a
    trailing comment.
- `prek.toml` hook revs.
- The OpenTofu providers of the telemetry infrastructure in `telemetry/infra/`.
- The wrangler version that `deploy-telemetry.yml` pins.

!!! note "Why prek hooks need a custom manager"

    Renovate has no native prek manager yet
    ([renovatebot/renovate#41683](https://github.com/renovatebot/renovate/issues/41683)). Its built-in
    pre-commit manager matches only `.pre-commit-config.ya?ml`. This repository uses `prek.toml`, so a
    regex `customManagers` entry tracks the hook revs instead. This entry reads the `repo` / `rev` pairs
    and resolves them through the `github-tags` datasource. When the native manager ships, replace this
    entry with it.

### Range strategy

Renovate uses `rangeStrategy = "bump"`. With the default `auto` strategy, an open `>=` range still
covers a new release. In that case, only `uv.lock` changes, and the declared floor in
`pyproject.toml` goes stale. `bump` instead raises the lower bound (for example `cyclopts>=4.16.1`
to `cyclopts>=4.17.0`) and relocks. As a result, the declared floor tracks the version that is
actually installed. The github-actions manager and the prek.toml regex manager ignore
`rangeStrategy`, because they pin exact SHAs and revs, not version ranges.

### Schedule and release age

PRs open in a weekly window: Monday 02:00-06:59 in the `Europe/Berlin` timezone. The window only
allows PRs to open. The hosted Renovate App decides the actual time of each run.

Renovate holds non-security releases for 1 day (`minimumReleaseAge: "1 days"`) before it opens a PR.
Security fixes do not wait. `renovate.json5` enables both `vulnerabilityAlerts` and
`osvVulnerabilityAlerts`, and the security rule sets `minimumReleaseAge: "0 days"`. As a result,
those PRs open immediately and bypass both the weekly window and the 1-day hold.

!!! note

    The 1-day hold matches `[tool.uv] exclude-newer = "1 day"`. uv refuses to lock an artifact younger
    than a day. If Renovate proposes a release that is less than a day old, `uv lock` cannot resolve it,
    and the PR is broken. The two windows are the same length, so the hold leaves no margin. Nobody
    merges these PRs automatically, so a person reviews each one.

### PR titles and grouping

`renovate.json5` enables semantic commits with a `build` type and a `deps` scope. As a result, PR
titles read `build(deps): ...`.

Renovate groups updates to keep each review focused. It batches minor and patch updates per
ecosystem. Major updates land alone, so each one gets its own review:

| Group                              | What it batches                                              |
| ---------------------------------- | ------------------------------------------------------------ |
| `github-actions (minor+patch)`     | GitHub Actions minor and patch bumps                         |
| `python production (minor+patch)`  | `[project.dependencies]` minor and patch bumps               |
| `python development (minor+patch)` | `[dependency-groups]` (dev and test) minor and patch bumps   |
| `prek hooks`                       | Remaining prek hook revs without a cross-file twin           |
| `ruff`                             | The `ruff` PyPI dep and the `astral-sh/ruff-pre-commit` hook |
| `bandit`                           | The `bandit` PyPI dep and the `PyCQA/bandit` hook            |

The `ruff` and `bandit` groups have no update-type filter on purpose. For each of these tools, the
pyproject dependency and the prek hook must move together on every bump, majors included. As a
result, the dependency and the hook of each tool converge to one version in a single PR.

### Validating the config

Run the official validator to make sure that a change to `renovate.json5` is valid:

```bash
npx --yes --package renovate -- renovate-config-validator --strict
```

The same command runs as a pre-commit hook. As a result, the hook catches a malformed configuration
before it reaches CI.

## Version sync

The package version appears in a few files at once. The `check-version-sync` pre-commit hook keeps
the version the same in `pyproject.toml`, `uv.lock`, `SECURITY.md`, and `CHANGELOG.md`. As a result,
a release bump cannot land in one file and drift in another. The Python package reads its own
version from the installed distribution metadata (`importlib.metadata.version`). As a result, the
source has no version string to keep in sync. For the rest of the local checks that run before a
commit, see [Code quality](code-quality.md).
