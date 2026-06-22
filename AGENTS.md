# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Architecture

`photo-tagger` is a single CLI package under `src/photo_tagger/`. The flow is: discover files → read
existing metadata → build a prompt → call a vision model → merge keywords → write XMP/IPTC. Each
module owns one slice of that pipeline:

- `main.py` - cyclopts entry point and orchestration (`tag` default command, `doctor` command). Keep
  it thin: it wires modules together and translates domain errors into exit codes.
- `cli_options.py` - the CLI *schema*: the seven `@dataclass` option groups, their config-file
  defaults (`load_defaults`), and the `ProcessingOptions` mapping. New flags go here, not in `main`.
- `providers.py` - the backend registry. A `ProviderBackend` (Strategy) bundles the per-backend bits
  (listing URL, listing parser, provider factory); shared HTTP plumbing is written once. Add a
  backend by adding one entry to `_BACKENDS` plus its defaults in `config.py`, and a name to the
  `ProviderName` Literal (a test asserts the two stay in sync).
- `ai.py` - builds the pydantic-ai `Agent` from a backend and runs inference.
- `pipeline.py` - the batch runner (serial + thread-pool passes, retry pass, usage accounting).
- `metadata.py` - all ExifTool reads/writes; `keywords.py` - hierarchical-keyword merge logic.
- `models.py` - shared data types: `GeneratedMetadata` (the model's schema), `InferenceResult`, and
  `KeywordSet` (the typed value object that replaced the old `dict[str, list[str]]`).
- `cache.py`, `locking.py`, `image_io.py`, `discovery.py`, `progress.py`, `logging_setup.py`,
  `diagnostics.py` - focused single-purpose helpers; `errors.py` - the domain exception hierarchy.
- `gui.py` - optional PySide6 desktop frontend (the `photo-tagger gui` command), a thin shell over
  `run_batch`; `gui_state.py` - its Qt-free, unit-tested logic. The GUI is additive: no pipeline
  code is Qt-aware. PySide6 lives behind the `[gui]` extra and is imported lazily.

The version is single-sourced: `photo_tagger.__version__` reads installed distribution metadata, so
`pyproject.toml`'s `[project].version` is the only code-side source. Do not reintroduce a hardcoded
version string. `scripts/check_version_sync.py` guards the hand-maintained docs against it.

## Code style

- Python 3.14: use modern syntax. PEP 695 generics, `match`, structural pattern matching, and PEP
  758 `except` without parentheses (`except ValueError, TypeError:`) are all fine.
- 100-column width for code, comments, and docstrings.
- Write comments and docstrings in plain, simple language. Explain the _why_, not the _what_.
- Never use em-dash (`—`). Use a regular hyphen or rephrase.

## Tests and code quality

- Add or update tests whenever you change behavior. Target: 90%+ coverage.
- `uv run pytest` (full suite) or `uv run pytest -k test_name` (single test).
- Keep functions small and focused. If a function grows complex (deep nesting, many branches, hard
  to test), refactor before adding more to it. Extract helpers, split responsibilities.
- SonarQube has many false positives. Silence with `# NOSONAR` (and a one-line reason) only when the
  finding clearly does not apply. Default is to fix, not silence.
- **GUI code is special.** Put testable GUI logic in `gui_state.py` (plain Python, covered
  normally). `gui.py` is the Qt shell: it is excluded from coverage (`[tool.coverage.run].omit` plus
  `sonar.coverage.exclusions`, so SonarQube does not count its untested-in-CI lines against new-code
  coverage), from zuban via its `# mypy: ignore-errors` header, and from pycroscope via a module
  override (shiboken generates Qt attributes at runtime, so static tools see false
  `undefined_attribute` errors). `tests/test_gui.py` carries the same `# mypy: ignore-errors`
  header: it imports PySide6, which has no stubs and is not installed in the lint job, so zuban
  would otherwise only report an unresolved import. Test it with `uv sync --extra gui --group test`
  then `QT_QPA_PLATFORM=offscreen uv run pytest tests/test_gui.py`; those tests `importorskip`
  PySide6 so the suite stays green without the extra.

## Before committing

```bash
uv run ruff check --fix .                              # Lint + auto-fix
uv run ruff format .                                   # Format
zuban check                                            # Strict type check (mypy-compatible)
uv run pycroscope --config-file pyproject.toml         # Semi-static analyzer (complements zuban)
prek run -a                                            # Run all pre-commit hooks
```

`pycroscope` config lives under `[tool.pycroscope]` in `pyproject.toml`. It does **not**
auto-discover the config file, so always pass `--config-file pyproject.toml` when invoking it
directly. The pre-commit hook already does this.

The full local CI + SonarQube refresh pipeline lives in the `/ci` skill
([.claude/skills/ci.md](.claude/skills/ci.md)). Claude **may** invoke it, but should do so
sparingly:

- Each run takes ~1-2 minutes and uploads to SonarCloud, so treat it as a verification step, not a
  loop.
- Run it at most twice per round of fixes, to confirm the findings actually cleared. Do not re-run
  just because the `mcp__sonarqube__*` tools still show yesterday's data; wait until there is a real
  set of changes to verify.

The `mcp__sonarqube__*` tools query findings from the last upload.

Two MCP servers are configured in `.mcp.json`: `sonarqube` (self-hosted Docker, full toolset) and
`sonarqubeCloud` (cloud-native HTTP, smaller toolset). Default is `sonarqube`. Only switch to
`sonarqubeCloud` if the Docker-based server is unavailable (e.g., Docker not running).

## Git

- Make small, atomic commits. One logical change per commit.
- **Commit as you go.** After finishing a discrete change (a single fix, one refactor, one file's
  worth of related edits), run the pre-commit checks and create the commit before moving to the next
  change. Do not batch many unrelated edits into one large commit at the end of a session.
- Write clear commit messages: short imperative subject, body describing the _why_.

## Things Claude must never do

- **Never push to GitHub** unless the user explicitly asks in this turn. Creating local commits is
  fine; `git push` is not.
- **Never skip hooks** (`--no-verify`, `--no-gpg-sign`). If a hook fails, fix the cause.
- **gpg-agent restart is pre-authorized.** If a commit hangs, times out, or fails with a pinentry /
  gpg-agent error (common after the laptop wakes from sleep), Claude MUST run
  `gpgconf --kill gpg-agent` itself via the Bash tool and retry the commit. Do not stop and wait for
  the user: this command is the standard fix, safe to run at any time, and the commit cannot
  complete until the agent is restarted.
- **Never commit real secrets.** If `django-vars.env` or `docker/.env` shows non-placeholder values
  in a staged diff, stop and flag it.
- **Never silence lint/type/SonarQube findings just to make CI pass.** Fix the underlying issue;
  silence only with a documented reason.
- **Never rewrite shared history** (`git rebase`, `git reset --hard`, force-push) on any branch that
  has been pushed, unless the user asks.

## Context management

When compacting, preserve: the list of files modified in this session, any failing tests or open
SonarQube findings not yet resolved, and any pending todos. Drop large file dumps and exploratory
search output.
