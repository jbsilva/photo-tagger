# CLAUDE.md

This file gives guidance to Claude Code (claude.ai/code) for work in this repository.

## Architecture

`photo-tagger` is a single CLI package under `src/photo_tagger/`. The flow is: discover files → read
existing metadata → build a prompt → call a vision model → merge keywords → write XMP/IPTC. Each
module owns one part of that pipeline:

- `main.py`: the cyclopts entry point and the orchestration (the default `tag` command, plus the
  `doctor`, `watch`, and `undo` commands). Keep it thin. It connects the modules and translates
  domain errors into exit codes. `_RunSetup` holds what a run builds once (agent, cache, report
  writers, undo journal). `_process_batch` runs one batch against it, so `tag` and `watch` share one
  path.
- `cli_options.py`: the CLI _schema_. It holds the seven `@dataclass` option groups, their defaults
  from the configuration file (`load_defaults`), and the `ProcessingOptions` mapping. New flags go
  here, not in `main`.
- `providers.py`: the backend registry. A `ProviderBackend` (Strategy) bundles the parts that differ
  per backend: the listing URL, the listing parser, and the provider factory. The shared HTTP code
  exists once. To add a backend, add one entry to `_BACKENDS`, its defaults in `config.py`, and a
  name to the `ProviderName` Literal. A test makes sure that the two stay in sync.
- `ai.py`: builds the pydantic-ai `Agent` from a backend and runs inference.
- `pipeline.py`: the batch runner (serial and thread-pool passes, the retry pass, usage accounting).
- `metadata.py`: all ExifTool reads and writes. `write_plan.py` decides which files a save writes
  (the photo, its sidecar, or both) and records what undo needs about each. `prompt_context.py`
  turns what a photo already carries into the prompt's metadata section. `keywords.py`: the merge
  logic for hierarchical keywords.
- `vocabulary.py`: the controlled vocabulary (`--vocabulary`). It parses a keyword list and snaps
  generated keywords onto it. `sessions.py` groups a batch into shoots (`--session-gap`) and builds
  the vocabulary of each shoot from the output of that shoot. Both go through the same matcher.
- `script_guard.py`: flags generated text that mixes writing systems (a CJK word in English output).
  It is pure and knows no languages. It reads the dominant script from the text itself and does not
  map `--output-language` to a script. Thus callers run it again on cache replays, and the cache
  stores no verdict. `--repair-language` then asks the model again, with the prompt in
  `config.build_repair_prompt`. `language_repair.py` holds that repair once for both frontends: each
  passes the one model call it makes, and gets back the answer to keep and a `RepairStatus`. The
  repair runs before the cache write, so a rerun never pays for a repair that worked. One that came
  back mixed leaves the original cached and is asked again.
- `fields.py`: the copyright and credit fields a run writes (`ExtraFields`), and `--set`.
  `templates.py` parses and fills their notice templates: chains of tags with quoted fallbacks. It
  is pure: the tag values arrive as a mapping, read before the write.
- `undo.py`: the write journal of each run and the restore logic of the `undo` command. `watch.py`:
  the folder watcher (it polls) behind the `watch` command.
- `models.py`: shared data types. `GeneratedMetadata` is the schema of the model, next to
  `InferenceResult` and `KeywordSet` (the typed value object that replaced the old
  `dict[str, list[str]]`).
- `telemetry.py`: the anonymous usage beacons. It holds the consent logic, the event dataclasses
  (the closed list of what is sent), and the sending. `telemetry_facts.py` reduces URLs, install
  paths, stack frames, and the environment to coarse categories for it. A new field needs a line in
  `docs/telemetry.md`.
- `cache.py`, `locking.py`, `image_io.py`, `discovery.py`, `progress.py`, `logging_setup.py`,
  `diagnostics.py`, `programs.py`: small helpers with one purpose each. `programs.py` finds every
  helper program (ExifTool too) by absolute path, never in the current directory. `errors.py`: the
  hierarchy of domain exceptions.
- `gui/`: the optional PySide6 desktop frontend (the `photo-tagger gui` command). It reviews before
  it writes, and runs `create_agent` and `analyze_image_with_ai` on threads of its own, not
  `run_batch`. `gui/main_window.py` holds the window, `gui/workers.py` holds the background jobs,
  and `gui/app.py` holds `launch`. Each part of the window has a module of its own: a dialog, a
  menu, or a pane such as `detail_pane.py` or `folder_grid.py`. `gui_state.py` holds its Qt-free
  logic, with unit tests. `gui_usage.py` holds its Qt-free usage tracking for telemetry (runs,
  sessions, busy and idle time, edits before a save). The GUI only adds to the pipeline: no pipeline
  code knows about Qt. PySide6 is in the `[gui]` extra, and the code imports it lazily.
- `telemetry/` (outside the package): the collector. It holds the Cloudflare Worker, its D1
  migrations, the marimo dashboard, and the OpenTofu configuration for the Cloudflare resources in
  `telemetry/infra/`. See `telemetry/README.md`.

The version has a single source. `photo_tagger.__version__` reads the metadata of the installed
distribution, so `[project].version` in `pyproject.toml` is the only source in the code. Do not add
a hardcoded version string again. `scripts/check_version_sync.py` makes sure that the docs that
people maintain by hand agree with it.

## Code style

- Python 3.14: use modern syntax. PEP 695 generics, `match`, structural pattern matching, and PEP
  758 `except` without parentheses (`except ValueError, TypeError:`) are all fine.
- Keep code, comments, and docstrings within 100 columns.
- Write comments and docstrings in plain, simple language. Explain the _why_, not the _what_.
- Never use an em-dash (`—`) or an en-dash (`–`). Rephrase instead: use a colon, a semicolon,
  parentheses, or two sentences. A spaced hyphen is not a substitute, because it means something
  else. Hyphens are for compound words (`event-scoped`) and ranges (`09:00 - 17:30`).

## Tests and code quality

- Add or update tests each time you change behavior. The target is a coverage of 90% or more.

- Run `uv run poe test` for the full suite, or `uv run poe test -k test_name` for a single test.
  Extra arguments go to pytest.

- Keep functions small and focused. If a function becomes complex (deep nesting, many branches, hard
  to test), refactor it before you add more to it. Extract helpers, and split responsibilities.

- SonarQube reports many false positives. If a finding clearly does not apply, silence it with
  `# NOSONAR` and a one-line reason. By default, fix the finding and do not silence it.

- Write the reason after a `# noqa` code without a comma. Ruff ignores the text, but SonarQube reads
  it as more codes and reports rule S7632. Example: `# noqa: PLR0913 - every kwarg is one knob.`

- GUI code has special rules. Put testable GUI logic in `gui_state.py` (plain Python, with normal
  coverage). The `gui/` package is the Qt shell, and three tools leave it out:

  - Coverage leaves it out through `[tool.coverage.run].omit` and `sonar.coverage.exclusions`. Thus
    SonarQube does not count its lines (untested in CI) against the coverage of new code.
  - zuban leaves it out through the `# mypy: ignore-errors` header of each module.
  - pycroscope leaves it out through a `# static analysis: ignore` header at the top of each module.
    pycroscope cannot import PySide6 in the lint job, which does not install the `[gui]` extra.
    Also, shiboken makes Qt attributes at runtime, so static tools misread the Qt shell in any case.

  A new module in `gui/` needs both headers. Copy them from an existing module.

  A part of the window reports back to the window through Qt signals: `status` for the status line,
  `used` for a feature count in telemetry, and a signal for each request. It asks the window for
  what it needs through callables that its constructor takes. The window owns the photos and the
  runs, and connects the parts to each other. A new part follows the same pattern. The tests patch a
  module-level name through `_patch_gui`, which patches every GUI module that imports it.

  `tests/test_gui.py` has the same `# mypy: ignore-errors` header. It imports PySide6, which has no
  stubs and is not installed in the lint job. Without the header, zuban reports only an unresolved
  import. To test the GUI, run `uv run poe test-gui`, which installs the extra and runs Qt
  offscreen. Those tests call `importorskip` for PySide6, so the suite stays green without the
  extra.

## Tasks

The development tasks are [Poe the Poet](https://poethepoet.natn.io) tasks under `[tool.poe]` in
`pyproject.toml`. `uv run poe` lists them. Use a task, and do not type its command again. If a new
command becomes part of the routine, add a task for it.

## Before committing

```bash
uv run poe check    # The six tasks below, in order; stops at the first failure
```

```bash
uv run poe lint         # ruff check --fix .
uv run poe format       # ruff format .
uv run poe typecheck    # zuban check (strict, mypy-compatible)
uv run poe pycroscope   # Semi-static analyzer (complements zuban)
uv run poe imports      # lint-imports (the layering contracts in [tool.importlinter])
uv run poe hooks        # prek run -a (all pre-commit hooks)
```

The configuration of `pycroscope` is under `[tool.pycroscope]` in `pyproject.toml`. pycroscope does
not find this file by itself. If you run it directly, always pass `--config-file pyproject.toml`.
The `pycroscope` task and the pre-commit hook already do this.

[`zizmor`](https://docs.zizmor.sh) (`uv run zizmor .`) audits everything under `.github/workflows/`.
It runs as a pre-commit hook and in `zizmor.yml`, which also uploads SARIF to the Security tab. A
workflow edit must keep the audit clean:

- Pin each action to a full commit SHA.
- Set `persist-credentials: false` on every checkout.
- Use no caching in `publish.yml`.
- Give each job the narrowest `permissions`.

The `/ci` skill ([.claude/skills/ci/SKILL.md](.claude/skills/ci/SKILL.md)) holds the full local
pipeline for CI and the SonarQube refresh. Claude can run it, but not often:

- Each run takes about 1 to 2 minutes and uploads to SonarCloud. Run it to make sure that a set of
  changes works, not in a loop.
- Run it at most twice per round of fixes, to make sure that the findings are gone. Do not run it
  again only because the `mcp__sonarqube__*` tools still show old data. Wait until you have a real
  set of changes to test.

The `mcp__sonarqube__*` tools query the findings of the last upload.

`.mcp.json` configures two MCP servers: `sonarqube` (self-hosted Docker, full toolset) and
`sonarqubeCloud` (cloud-native HTTP, smaller toolset). The default is `sonarqube`. If the Docker
server is not available (for example, Docker is not running), switch to `sonarqubeCloud`. In all
other cases, use `sonarqube`.

## Git

- Make small, atomic commits: one logical change per commit.
- Commit as you go. After you finish a discrete change (a single fix, one refactor, the related
  edits in one file), run the pre-commit checks. Then make the commit, before you move to the next
  change. Do not collect many unrelated edits into one large commit at the end of a session.
- Write clear commit messages: a short imperative subject, and a body that explains the _why_.

## Things Claude must never do

- Never push to GitHub, unless the user asks for it explicitly in this turn. Local commits are fine.
  `git push` is not.
- Never skip hooks (`--no-verify`, `--no-gpg-sign`). If a hook fails, fix the cause.
- A restart of gpg-agent has approval in advance. If a commit hangs, times out, or fails with a
  pinentry or gpg-agent error (common after the laptop wakes from sleep), Claude MUST run
  `gpgconf --kill gpg-agent` itself with the Bash tool and retry the commit. Do not stop and wait
  for the user. This command is the standard fix, it is safe at any time, and the commit cannot
  complete until the agent restarts.
- Never commit real secrets. If `django-vars.env` or `docker/.env` shows values that are not
  placeholders in a staged diff, stop and tell the user.
- Never silence lint, type, or SonarQube findings only to make CI pass. Fix the cause. Silence a
  finding only with a documented reason.
- Never rewrite shared history (`git rebase`, `git reset --hard`, a force-push) on a branch that is
  on the remote, unless the user asks.

## Context management

At compaction, keep: the list of files that changed in this session, any failing tests, any open
SonarQube findings, and any pending todos. Drop large file dumps and the output of exploratory
searches.
