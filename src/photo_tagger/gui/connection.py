# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The header row that picks the model, and the Connection dialog behind it."""

from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from photo_tagger.diagnostics import run_checks
from photo_tagger.errors import ProviderError
from photo_tagger.gui.widgets import busy_cursor, close_row
from photo_tagger.gui_state import Connection, check_results_html, rank_vision_models, tooltip
from photo_tagger.i18n import _, ngettext
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES, ProviderName, get_backend


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from photo_tagger.cli_options import ProviderConfig


class ConnectionBar(QWidget):
    """
    Say which model a run talks to, and where it lives.

    One compact row holds the choice people change often (the provider and the model) and a button
    for the set-once settings (the server URL and the API key), which live in a dialog of their own
    with a connection test. Every job reads :meth:`connection` when it starts. The bar reports a
    status line with :attr:`status` and each feature used with :attr:`used`.
    """

    status = Signal(str)
    used = Signal(str)

    def __init__(self, parent: QWidget, provider: ProviderConfig) -> None:
        """Build the row and its dialog, pre-filled from the config file's provider settings."""
        super().__init__(parent)
        self._provider = QComboBox()
        for name in PROVIDER_NAMES:
            self._provider.addItem(PROVIDER_LABELS.get(name, name), name)
        self._provider.setCurrentIndex(max(0, list(PROVIDER_NAMES).index(provider.provider_name)))
        # Size to the widest label so "LM Studio" is not clipped.
        self._provider.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._provider.setMinimumContentsLength(10)
        self._provider.setToolTip(tooltip("Backend that serves the vision-language model."))

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setMinimumWidth(260)
        self._model.setCurrentText(provider.model_name)
        self._model.setToolTip(
            tooltip(
                "Model identifier. Type it, or press Refresh to list what the provider serves.",
            ),
        )
        refresh = QPushButton(_("Refresh"))
        refresh.setToolTip(tooltip("Query the provider for the models it currently serves."))
        refresh.clicked.connect(self.refresh_models)

        self._dialog = self._build_dialog(provider)
        connection = QPushButton(_("Connection..."))
        connection.setToolTip(tooltip("Server URL, API key, and a connection test."))
        connection.clicked.connect(lambda: (self.used.emit("connection"), self._dialog.exec()))

        row = QHBoxLayout(self)
        # Flush with the window's own margins, as the row was before it had a widget of its own.
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(_("Provider")))
        row.addWidget(self._provider)
        row.addWidget(QLabel(_("Model")))
        row.addWidget(self._model, stretch=1)
        row.addWidget(refresh)
        row.addWidget(connection)

    def _build_dialog(self, provider: ProviderConfig) -> QDialog:
        """URL, API key, and the connection test: set-once settings, out of the main window."""
        dialog = QDialog(self)
        dialog.setWindowTitle(_("Connection settings"))
        dialog.setMinimumWidth(520)
        form = QFormLayout(dialog)
        # macOS style defaults to fixed-size fields; let them fill the dialog width instead.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._url = QLineEdit(provider.api_base_url or "")
        self._url.setMinimumWidth(380)
        self._url.setPlaceholderText(_("(provider default URL)"))
        self._url.setToolTip(
            tooltip("Provider API base URL. Leave blank to use the provider's default."),
        )
        form.addRow(_("Base URL"), self._url)

        # Pre-filled from a config-file key if one is set, never from an environment variable: an
        # env key stays in the environment and is resolved at call time, so it never lands in the
        # widget. A typed key is masked, used only for this session, and never written to disk.
        self._api_key = QLineEdit(provider.api_key or "")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setClearButtonEnabled(True)
        self._api_key.setMinimumWidth(380)
        self._api_key.setPlaceholderText(_("(uses provider env var)"))
        self._api_key.setToolTip(
            tooltip(
                "API key for the provider. Leave blank to use the provider's environment variable "
                "(OPENAI_API_KEY, LM_STUDIO_API_KEY, LLAMA_CPP_API_KEY, or OLLAMA_API_KEY). "
                "Required for OpenAI. A typed key is used for this session only and is never "
                "written to disk.",
            ),
        )
        form.addRow(_("API key"), self._api_key)

        self._test_button = QPushButton(_("Test Connection"))
        self._test_button.setToolTip(
            tooltip("Check ExifTool and that the provider serves the model."),
        )
        self._test_button.clicked.connect(self.test_connection)
        form.addRow(close_row(dialog, self._test_button))
        return dialog

    def connection(self) -> Connection:
        """Read the provider, model, server, and key off the row and the dialog."""
        return Connection(
            # The combo shows a label and carries the internal name as item data.
            provider=cast("ProviderName", self._provider.currentData()),
            model=self._model.currentText().strip(),
            api_base_url=self._url.text().strip() or None,
            # None falls back to the provider's environment variable, then its default.
            api_key=self._api_key.text().strip() or None,
        )

    def set_testable(self, *, enabled: bool) -> None:
        """Offer the connection test, or hold it back while a run is using the connection."""
        self._test_button.setEnabled(enabled)

    def refresh_models(self) -> None:
        """Ask the provider for the models it serves and list the likely vision models first."""
        self.used.emit("refresh_models")
        connection = self.connection()
        backend = get_backend(connection.provider)
        base_url = connection.api_base_url or backend.default_base_url
        try:
            with busy_cursor():
                models = backend.list_models(base_url, backend.resolve_api_key(connection.api_key))
        except ProviderError as exc:
            QMessageBox.warning(self.window(), _("Could not list models"), str(exc))
            return
        current = self._model.currentText()
        self._model.clear()
        self._model.addItems(rank_vision_models(models))
        self._model.setCurrentText(current)
        self.status.emit(
            ngettext(
                "Found {n} model on {provider}.",
                "Found {n} models on {provider}.",
                len(models),
            ).format(n=len(models), provider=PROVIDER_LABELS.get(connection.provider, "")),
        )

    def test_connection(self) -> None:
        """Check ExifTool and the provider, and show what passed and what did not."""
        self.used.emit("test_connection")
        connection = self.connection()
        with busy_cursor():
            results = run_checks(
                connection.provider,
                connection.model,
                api_base_url=connection.api_base_url,
                api_key=connection.api_key,
            )
        box = QMessageBox(self.window())
        box.setWindowTitle(_("Connection check"))
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(check_results_html(results))
        all_ok = all(r.ok for r in results)
        box.setIcon(QMessageBox.Icon.Information if all_ok else QMessageBox.Icon.Warning)
        box.exec()
