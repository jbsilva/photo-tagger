"""Keep the import-linter contracts in step with the modules they guard."""

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "src" / "photo_tagger"
# The frontends: the core may not import them, so they are left out of its contract on purpose.
FRONTENDS = {"cli_options", "cli_reports", "cli_telemetry", "gui", "main"}


def _contracts() -> dict[str, dict[str, object]]:
    """Return the import-linter contracts in pyproject.toml, keyed by name."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = config["tool"]["importlinter"]["contracts"]
    return {contract["name"]: contract for contract in contracts}


def _modules() -> set[str]:
    """Name every top-level module of the package, the gui package included."""
    names = {path.stem for path in PACKAGE.glob("*.py") if not path.stem.startswith("_")}
    return names | {path.name for path in PACKAGE.iterdir() if (path / "__init__.py").exists()}


def _listed(contract: dict[str, object]) -> set[str]:
    sources = contract["source_modules"]
    assert isinstance(sources, list)
    return {str(name).removeprefix("photo_tagger.") for name in sources}


def test_the_core_contract_covers_every_core_module() -> None:
    """A new module joins the core's list, or the contract silently stops covering it."""
    contract = _contracts()["The core knows neither frontend"]

    assert _listed(contract) == _modules() - FRONTENDS


def test_the_qt_contract_covers_every_module_but_the_gui_package() -> None:
    """Only the gui package may import Qt, so every other module has to be listed."""
    contract = _contracts()["Only the gui package imports Qt"]

    assert _listed(contract) == _modules() - {"gui"}


def test_the_architecture_page_lists_every_module() -> None:
    """The module table is the map a contributor reads first, so a new module needs a row."""
    page = (ROOT / "docs" / "architecture" / "index.md").read_text(encoding="utf-8")
    rows = set(re.findall(r"^\| \[`(\w+)(?:\.py|/)`\]", page, re.MULTILINE))

    assert rows == _modules()
