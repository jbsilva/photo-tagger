---
icon: lucide/package
---

# Dependencies

[uv](https://docs.astral.sh/uv/) manages the project's virtual environment, and `uv.lock` is
committed so every contributor and CI run builds against the exact same pinned set of packages.

## uv and the lockfile

`uv.lock` records the resolved version of every direct and transitive dependency. Because it is
checked in, `uv sync` reproduces an identical environment everywhere, and Renovate edits the lock in
lockstep with `pyproject.toml`.

To keep resolution reproducible over time, `pyproject.toml` pins how fresh an artifact may be:

```toml
[tool.uv]
exclude-newer = "1 day"
```

`exclude-newer = "1 day"` tells uv to resolve only against packages published at least a day ago.
Brand-new releases are invisible to `uv lock` until they have aged past that window, which avoids
locking onto an artifact that was just published and may still be yanked or repaired.

## Renovate

[`renovate.json5`](https://github.com/jbsilva/photo-tagger/blob/main/renovate.json5) drives
automated dependency updates and replaces the old `.github/dependabot.yml`. It extends
`config:best-practices` and `:enablePreCommit`, and covers three update domains:

- `pyproject.toml` plus `uv.lock` (the Python dependencies).
- GitHub Actions in `.github/workflows/`, pinned to a full commit SHA with the version kept in a
    trailing comment.
- `.pre-commit-config.yaml` hook revs.

### Range strategy

Renovate uses `rangeStrategy = "bump"`. With the default `auto` strategy an open `>=` range still
covers a new release, so only `uv.lock` would change and the declared floor in `pyproject.toml`
would go stale. `bump` instead raises the lower bound (for example `cyclopts>=4.16.1` to
`cyclopts>=4.17.0`) and relocks, so the declared floor tracks what is actually installed. The
github-actions and pre-commit managers ignore this setting because they pin exact SHAs and revs
rather than version ranges.

### Schedule and release age

PRs open in a weekly window: Monday 02:00-06:59 in the `Europe/Berlin` timezone. The window only
allows PRs to open; the hosted Renovate App decides when it actually runs.

Non-security releases are held for 3 days (`minimumReleaseAge: "3 days"`) before a PR is opened.
Security fixes do not wait: both `vulnerabilityAlerts` and `osvVulnerabilityAlerts` are enabled, and
the security rule pins `minimumReleaseAge: "0 days"`, so those PRs open immediately and bypass both
the weekly window and the 3-day hold.

!!! note

    The 3-day cooldown is also what keeps Renovate compatible with `[tool.uv] exclude-newer = "1 day"`.
    uv refuses to lock an artifact younger than a day, so if Renovate proposed a release that fresh, the
    resulting `uv lock` could not resolve and the PR would be broken. Holding non-security releases for
    3 days keeps every proposed version comfortably past uv's window.

### PR titles and grouping

Semantic commits are enabled with a `build` type and `deps` scope, so PR titles read
`build(deps): ...`.

Updates are grouped to keep review focused. Minor and patch updates are batched per ecosystem, while
major updates land alone so each gets its own review:

| Group                              | What it batches                                              |
| ---------------------------------- | ------------------------------------------------------------ |
| `github-actions (minor+patch)`     | GitHub Actions minor and patch bumps                         |
| `python production (minor+patch)`  | `[project.dependencies]` minor and patch bumps               |
| `python development (minor+patch)` | `[dependency-groups]` (dev and test) minor and patch bumps   |
| `pre-commit hooks`                 | Remaining pre-commit hook revs without a cross-file twin     |
| `ruff`                             | The `ruff` PyPI dep and the `astral-sh/ruff-pre-commit` hook |
| `bandit`                           | The `bandit` PyPI dep and the `PyCQA/bandit` hook            |

The `ruff` and `bandit` groups deliberately have no update-type filter: the tool's pyproject
dependency and its pre-commit hook must move together on every bump, majors included, so each tool's
dep and hook converge to one version in a single PR.

### Validating the config

Validate any change to `renovate.json5` with the official validator:

```bash
npx --yes --package renovate -- renovate-config-validator --strict
```

The same command runs as a pre-commit hook, so a malformed config is caught before it reaches CI.

## Version sync

The package version appears in a few files at once. The `check-version-sync` pre-commit hook keeps
them aligned across `pyproject.toml`, `uv.lock`, `SECURITY.md`, and `CHANGELOG.md`, so a release
bump cannot land in one place and drift in another. The Python package reads its own version from
the installed distribution metadata (`importlib.metadata.version`), so there is no version string in
the source to keep in sync. For the rest of the local checks that run before a commit, see
[Code quality](code-quality.md).
