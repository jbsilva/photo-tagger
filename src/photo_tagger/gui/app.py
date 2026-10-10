# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""
Start the desktop app: the environment, logging, and translations, then the window.

Everything here runs once per process, before (or around) the Qt event loop.
"""

import os
import sys
import time

from loguru import logger
from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
from PySide6.QtWidgets import QApplication

from photo_tagger import i18n, telemetry
from photo_tagger.cli_options import load_defaults
from photo_tagger.config_file import load_config
from photo_tagger.errors import ObservabilityError
from photo_tagger.gui.common import LOG_FOLDER
from photo_tagger.gui.main_window import MainWindow
from photo_tagger.gui.style import app_icon, stylesheet
from photo_tagger.gui_state import ensure_path_dirs, login_shell_path
from photo_tagger.logging_setup import setup_logging
from photo_tagger.observability import configure_tracing


def launch(argv: list[str] | None = None) -> int:
    """Create the application, show the main window, and run the event loop."""
    started = time.monotonic()
    # A Finder/Dock launch inherits a minimal PATH, so graft on the PATH the user's login shell
    # would set. That recovers exiftool wherever their package manager put it (Homebrew, Nix,
    # MacPorts, ...). A no-op for a normal shell launch, where those dirs are already present.
    os.environ["PATH"] = ensure_path_dirs(os.environ.get("PATH", ""), login_shell_path())
    # And bridge a config-file exiftool path into the env var metadata reads (an exported var wins).
    if (exiftool_path := load_defaults().exiftool_path) is not None:
        os.environ.setdefault("PHOTO_TAGGER_EXIFTOOL", exiftool_path)
    # File-only logging: the window carries the live status, so the terminal stays quiet, but a
    # durable log (with full tracebacks for failed photos) is written for the "Open logs" button.
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=LOG_FOLDER)
    # The GUI has no command line to pass --otel on, so tracing is configured from the [log] table
    # of the config file alone. A misconfigured endpoint must not stop the window from opening: the
    # reason is logged and the app runs untraced, unlike the CLI, where the user asked in this
    # breath and deserves to be told the run will not be observed.
    log_defaults = load_defaults().log
    try:
        configure_tracing(
            enabled=log_defaults.otel,
            endpoint=log_defaults.otel_endpoint,
            include_content=log_defaults.otel_content,
        )
    except ObservabilityError as exc:
        logger.warning("otel_setup_failed", error=str(exc))
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Photo Tagger")
    app.setApplicationDisplayName("Photo Tagger")
    app.setWindowIcon(app_icon())
    app.setStyleSheet(stylesheet())
    # Resolve the UI language before any widget is built: strings are baked at construction.
    # Qt's system locale is the hint of last resort; it knows the OS language even when no LANG
    # is exported (a Finder/Dock launch).
    configured = load_config().get("language")
    language = i18n.activate(
        str(configured) if configured else None,
        system_hint=QLocale.system().name(),
    )
    # A Finder launch has no LANG, but Qt still knows the OS language for the beacons.
    telemetry.set_locale_hint(QLocale.system().name())
    if language != "en":
        # Also translate Qt's own stock strings (file dialogs, standard buttons). Best-effort:
        # PySide6 wheels may not ship every qtbase catalog.
        qt_translator = QTranslator(app)
        translations_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if qt_translator.load(QLocale(language), "qtbase", "_", translations_dir):
            app.installTranslator(qt_translator)

    def _report_crash(exc: BaseException, *, block: bool) -> None:
        # The persisted toggle (or the config default) decides, as it does for the window itself;
        # emit_crash enforces the environment switches on top.
        enabled, opted_in = telemetry.gui_choice(config_enabled=load_defaults().telemetry.enabled)
        if telemetry.notice_due(interface="gui"):
            # The window has not shown its disclosure yet, so nothing may be sent before it has.
            return
        telemetry.emit_crash(
            exc,
            interface="gui",
            command="gui",
            enabled=enabled,
            opted_in=opted_in,
            block=block,
        )

    # Qt swallows exceptions raised inside slots (they reach sys.excepthook and the loop keeps
    # running), so an unhandled slot crash never propagates out of app.exec(). Chain a hook that
    # fires an anonymous crash beacon (type and stack trace, never the message) first.
    previous_hook = sys.excepthook

    def _crash_hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        _report_crash(exc, block=False)
        previous_hook(exc_type, exc, tb)  # type: ignore[arg-type]

    sys.excepthook = _crash_hook

    try:
        window = MainWindow()
        window.show()
        startup_seconds = time.monotonic() - started
        window.maybe_show_telemetry_notice()
        window.report_launch(startup_seconds)
        return app.exec()
    except Exception as exc:
        # A crash outside the event loop (startup, teardown) kills the process; flush the beacon.
        _report_crash(exc, block=True)
        raise
