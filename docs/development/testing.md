---
icon: lucide/flask-conical
---

# Testing

photo-tagger uses pytest with a small set of plugins configured up front, so a plain `uv run pytest`
already runs the suite with branch coverage, parallelism, and randomized order. This page explains
how to run the tests and what that configured setup gives you.

## Running the suite

Run the full test suite from the project root:

```bash
uv run pytest
```

To run a single test by name, pass a substring match to `-k`. pytest selects every test whose name
contains the value:

```bash
uv run pytest -k test_name
```

!!! tip

    While iterating on one change, run only the test you care about with `-k`. It skips the rest of the
    suite (and the parallel startup cost), so the feedback loop stays fast. Run the full `uv run pytest`
    again before you commit.

## What the configured setup does

The plugins below are wired into the default pytest run, so you get their behavior without extra
flags.

| Plugin                | What it adds                                                           |
| --------------------- | ---------------------------------------------------------------------- |
| `pytest-cov`          | Branch coverage, with `term-missing` output and `reports/coverage.xml` |
| `pytest-xdist`        | Parallel execution across CPUs via `--numprocesses=logical`            |
| `pytest-random-order` | Randomized test order each run                                         |
| `inline-snapshot`     | Inline snapshot assertions stored next to the test code                |

### Branch coverage

Coverage runs in branch mode through `pytest-cov`. The terminal report uses `term-missing`, so it
lists the line (and branch) numbers that were not exercised, and a machine-readable report is
written to `reports/coverage.xml` for tooling such as SonarCloud to pick up.

### Parallelism

`pytest-xdist` runs tests in parallel with `--numprocesses=logical`, which spreads the suite across
all logical CPUs. This is why test order should never be assumed: tests must not depend on each
other or on shared mutable state.

### Randomized order

`pytest-random-order` shuffles the test order on every run. This surfaces hidden ordering
dependencies between tests, where one test only passes because an earlier one left some state
behind.

### Inline snapshots

`inline-snapshot` keeps expected values inline in the test source rather than in separate fixture
files. When an expected value changes intentionally, the snapshot can be updated in place instead of
being hand-edited.

## Integration tests

Integration tests are marked with the `integration` marker and are deselected by default, so they do
not run during a normal `uv run pytest`. Run them explicitly with `-m`:

```bash
uv run pytest -m integration
```

!!! note

    Integration tests can reach out to external dependencies (such as a running model server or the
    `exiftool` binary), which is why they are kept out of the default run. See
    [Installation](../getting-started/installation.md) for the required tools.

## Coverage target

Aim for 90%+ coverage. Add or update tests whenever you change behavior so the suite keeps tracking
what the code actually does. For the lint, type-check, and analysis steps that run alongside the
tests, see [Code quality](code-quality.md).
