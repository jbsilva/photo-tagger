---
icon: lucide/flask-conical
---

# Testing

photo-tagger uses pytest with a small set of plugins that the project configures in advance. As a
result, a plain `uv run pytest` already runs the suite with branch coverage, parallelism, and
randomized order. This page explains how to run the tests and what that configured setup gives you.

## Running the suite

Run the full test suite from the project root:

```bash
uv run poe test
```

The `test` task gives any extra arguments to pytest. To run a single test by name, pass a substring
match to `-k`. pytest selects every test whose name contains the value:

```bash
uv run poe test -k test_name
```

!!! tip

    While you work on one change, run only the test that you care about with `-k`. It skips the rest of
    the suite (and the parallel startup cost), so the feedback loop stays fast. Run the full
    `uv run poe test` again before you commit.

## What the configured setup does

The project wires the plugins below into the default pytest run. As a result, you get their behavior
without extra flags.

| Plugin                | What it adds                                                           |
| --------------------- | ---------------------------------------------------------------------- |
| `pytest-cov`          | Branch coverage, with `term-missing` output and `reports/coverage.xml` |
| `pytest-xdist`        | Parallel execution across CPUs via `--numprocesses=logical`            |
| `pytest-random-order` | Randomized test order each run                                         |
| `inline-snapshot`     | Inline snapshot assertions stored next to the test code                |

### Branch coverage

Coverage runs in branch mode through `pytest-cov`. The terminal report uses `term-missing`, so it
lists the line (and branch) numbers that no test exercised. `pytest-cov` also writes a
machine-readable report to `reports/coverage.xml`. Tools such as SonarCloud pick up that report.

### Parallelism

`pytest-xdist` runs tests in parallel with `--numprocesses=logical`. This option spreads the suite
across all logical CPUs. For this reason, never assume a test order. Tests must not depend on each
other or on shared mutable state.

### Randomized order

`pytest-random-order` shuffles the test order on every run. This shows hidden ordering dependencies
between tests. In such a dependency, one test passes only because an earlier test left some state
behind.

### Inline snapshots

`inline-snapshot` keeps expected values inline in the test source, not in separate fixture files. If
an expected value changes on purpose, you can update the snapshot in place instead of editing it by
hand.

## Integration tests

Integration tests carry the `integration` marker. The default run deselects them, so they do not run
during a normal `uv run poe test`. Run them explicitly with `-m`:

```bash
uv run poe test -m integration
```

!!! note

    Integration tests can use external dependencies, such as a running model server or the `exiftool`
    binary. For this reason, the default run leaves them out. See
    [Installation](../getting-started/installation.md) for the required tools.

## GUI tests

The project tests the optional desktop GUI in two layers. As a result, the suite stays green whether
or not you installed the `gui` extra:

- `tests/test_gui_state.py` covers the Qt-free logic in `gui_state.py`. It imports no PySide6, so it
    always runs and counts toward coverage like any other module.
- `tests/test_gui.py` exercises the real window and worker. It calls
    `pytest.importorskip("PySide6")`. If the extra is absent (the default CI run), pytest skips the
    whole file. It uses Qt's `offscreen` platform to build widgets without a display server.

To run the GUI tests, use the `test-gui` task. It installs the extra and points Qt at the offscreen
platform:

```bash
uv run poe test-gui
```

The `gui/` package itself is the Qt event-loop shell. The offscreen GUI tests do drive its widgets
and workers. But coverage cannot measure the event loop, and the runtime attributes from `shiboken`
confuse the static tools. For these reasons, `[tool.coverage.run].omit` in `pyproject.toml` excludes
the whole package from coverage. Keep testable logic in `gui_state.py` (covered), not in `gui/`
(omitted). See [Code quality](code-quality.md) for why the project also excludes `gui/` from the
type checker and static analyzer.

## Coverage target

Aim for 90%+ coverage. Each time you change behavior, add or update tests. This keeps the suite in
line with what the code actually does. For the lint, type-check, and analysis steps that run
together with the tests, see [Code quality](code-quality.md).
