# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Settings menu: the toggles every run and save follows, and the two languages."""

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QInputDialog, QMenu, QMessageBox, QWidget

from photo_tagger import i18n, telemetry
from photo_tagger.cli_options import resolve_notify
from photo_tagger.config import DEFAULT_OUTPUT_LANGUAGE
from photo_tagger.errors import ConfigFileError
from photo_tagger.gui.common import CONFIG_SAVE_ERROR_TITLE, config_target
from photo_tagger.gui_state import (
    OUTPUT_LANGUAGE_SUGGESTIONS,
    config_text_with_language,
    config_text_with_output_language,
    telemetry_toggle_note,
    tooltip,
    wrap_tooltip,
)
from photo_tagger.i18n import _, gettext_noop
from photo_tagger.notify import APP_NAME, notify


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from collections.abc import Callable
    from pathlib import Path

    from PySide6.QtCore import SignalInstance

    from photo_tagger.cli_options import Defaults


class SettingsMenu(QMenu):
    """
    Hold the choices that shape every run and every save, and the two languages.

    The toggles live here and are read when a job starts: the result cache, anonymous telemetry,
    recording saves for undo, rewriting mixed-language output, and the end-of-run alert. The two
    language submenus persist a choice to the config file as soon as it is made. The dialogs and
    the config-file actions this menu lists belong to the window, which it asks for them through
    signals; it reports a status line with :attr:`status` and each feature used with :attr:`used`.
    """

    used = Signal(str)
    status = Signal(str)
    # The metadata language changed and was saved; the vocabulary's plural folding depends on it.
    output_language_changed = Signal(str)
    keyword_rules_requested = Signal()
    copyright_requested = Signal()
    pacing_requested = Signal()
    save_defaults_requested = Signal()
    edit_config_requested = Signal()

    def __init__(
        self,
        parent: QWidget,
        *,
        raw_config: dict[str, Any],
        defaults: Defaults,
        cache_file: Path,
    ) -> None:
        """
        Build the menu on the config file's choices.

        *cache_file* is where cached results go, named in the cache toggle's tooltip.
        """
        super().__init__(_("Settings"), parent)
        self.setToolTipsVisible(True)
        self._cache_file = cache_file
        # The persisted UI language choice; "auto" means follow the OS locale. Changing it here
        # rewrites this key and takes effect on the next launch.
        self._language = str(raw_config.get("language", i18n.AUTO))
        # The language the model writes titles, descriptions, and keywords in. Distinct from the
        # UI language above; persisted under [inference] output_language, which the CLI shares.
        self._output_language = defaults.inference.output_language
        self._repair_language = defaults.inference.repair_language
        # Telemetry on/off: a persisted Settings-menu choice wins over the config-file default, and
        # a toggle turned on by hand is an opt-in that overrules DO_NOT_TRACK.
        self._telemetry_enabled, self._telemetry_opted_in = telemetry.gui_choice(
            config_enabled=defaults.telemetry.enabled,
        )
        self._build_toggles(defaults)
        self.addSeparator()
        self._add_opener(
            _("Keyword Rules..."),
            self.keyword_rules_requested,
            gettext_noop(
                "The controlled vocabulary generated keywords are snapped onto, and whether the "
                "keywords of one shoot are made to agree with each other.",
            ),
        )
        self._add_opener(
            _("Copyright..."),
            self.copyright_requested,
            gettext_noop(
                "Your credit and copyright notice, written by every save. The year comes from "
                "each photo, so one setting covers every shoot.",
            ),
        )
        self._add_opener(
            _("Pacing..."),
            self.pacing_requested,
            gettext_noop(
                "Rest a long run now and then, so it does not flatten the battery, cook the "
                "machine, or take it over while you work.",
            ),
        )
        self.addSeparator()
        self._build_language_menu()
        self._build_output_language_menu()
        self.addSeparator()
        self._add_opener(
            _("Save Settings as Defaults..."),
            self.save_defaults_requested,
            gettext_noop(
                "Update the config file with the current provider, model, URL, file types, and "
                "save options. Other settings and comments in the file are preserved; the API "
                "key is never written.",
            ),
        )
        self._add_opener(
            _("Edit Config File..."),
            self.edit_config_requested,
            gettext_noop(
                "Open the config file in your default editor for the settings the GUI does not "
                "surface (prompt file, sampling, workers, filters, ...). Created if missing.",
            ),
        )

    def _add_opener(self, label: str, request: SignalInstance, hint: str) -> None:
        """Add an entry that asks the window for something: a dialog, or a config-file action."""
        action = self.addAction(label)
        action.triggered.connect(request)
        action.setToolTip(tooltip(hint))

    def _build_toggles(self, defaults: Defaults) -> None:
        """Add the on/off choices a run or a save reads when it starts."""
        self._cache_action = QAction(_("Cache AI Results"), self)
        self._cache_action.setCheckable(True)
        self._cache_action.setChecked(True)
        self._cache_action.setToolTip(
            tooltip(
                "Reuse earlier results for unchanged photos ({cache_file}). Uncheck to call "
                "the model again for everything; a single photo can skip the cache from its "
                "right-click menu.",
                cache_file=self._cache_file,
            ),
        )
        self._cache_action.toggled.connect(lambda _on: self.used.emit("cache_toggle"))
        self.addAction(self._cache_action)
        self._telemetry_action = QAction(_("Send Anonymous Telemetry"), self)
        self._telemetry_action.setCheckable(True)
        self._refresh_telemetry_action()
        self._telemetry_action.toggled.connect(self._on_telemetry_toggled)
        self.addAction(self._telemetry_action)
        self._undo_log_action = QAction(_("Record Saves for Undo"), self)
        self._undo_log_action.setCheckable(True)
        self._undo_log_action.setChecked(defaults.artifacts.undo_log)
        self._undo_log_action.setToolTip(
            tooltip(
                "Record every file a save writes, so Tools > Undo Writes can put it back. On by "
                "default: the saves worth undoing are the ones nobody planned to.",
            ),
        )
        self._undo_log_action.toggled.connect(lambda _on: self.used.emit("undo_log_toggle"))
        self.addAction(self._undo_log_action)
        self._build_repair_action()
        # End-of-run desktop alert, resolved the same way the CLI resolves its two flags: a config
        # that asks only for the sound is asking for the notification that carries it. The sound
        # box keeps its own saved value, greyed out while the alert is off, so turning the alert
        # back on brings the sound back too.
        notify_enabled, _sound = resolve_notify(defaults.display)
        self._build_notify_actions(enabled=notify_enabled, sound=defaults.display.notify_sound)

    def _build_repair_action(self) -> None:
        """Add the toggle that lets a run pay a second call to undo a mixed-script answer."""
        self._repair_action = QAction(_("Rewrite Mixed-Language Output"), self)
        self._repair_action.setCheckable(True)
        self._repair_action.setChecked(self._repair_language)
        self._repair_action.setToolTip(
            tooltip(
                "When the model answers in two writing systems, ask it again for the output "
                "language ('Ocarina' for a stray '陶笛'). Costs one extra call per affected "
                "photo. Off by default; those photos are marked either way.",
            ),
        )
        self._repair_action.toggled.connect(self._on_repair_toggled)
        self.addAction(self._repair_action)

    def _on_repair_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Record the repair choice so the next generate run and a settings save both see it."""
        self.used.emit("repair_toggle")
        self._repair_language = enabled

    def _build_notify_actions(self, *, enabled: bool, sound: bool) -> None:
        """Add the two end-of-run alert toggles, the sound one tied to the notification one."""
        self._notify_action = QAction(_("Notify When a Run Ends"), self)
        self._notify_action.setCheckable(True)
        self._notify_action.setChecked(enabled)
        self._notify_action.setToolTip(
            tooltip(
                "Raise a desktop notification when a generation or save run finishes, so a long "
                "batch can be started and left alone. Uses the notifier your system already has; "
                "nothing is shown if it has none.",
            ),
        )
        self._notify_action.toggled.connect(self._on_notify_toggled)
        self.addAction(self._notify_action)
        self._notify_sound_action = QAction(_("Play a Sound With Notifications"), self)
        self._notify_sound_action.setCheckable(True)
        self._notify_sound_action.setChecked(sound)
        self._notify_sound_action.setEnabled(enabled)
        self._notify_sound_action.setToolTip(
            tooltip("Play the system alert sound with the notification."),
        )
        self.addAction(self._notify_sound_action)

    def _on_notify_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Follow the notification toggle with its sound sub-toggle: no notification, no sound."""
        self.used.emit("notify_toggle")
        self._notify_sound_action.setEnabled(enabled)

    def _build_language_menu(self) -> None:
        """Add the Language submenu: System Default plus every shipped catalog."""
        self._language_menu = self.addMenu(_("Language"))
        self._language_menu.menuAction().setToolTip(
            tooltip(
                "Language of the app itself (menus, buttons, messages). The language of the "
                "generated metadata is set under Metadata Language.",
            ),
        )
        self._language_group = QActionGroup(self)
        entries = [(i18n.AUTO, _("System Default")), *i18n.SUPPORTED_LANGUAGES.items()]
        for code, label in entries:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(code == self._language)
            action.triggered.connect(
                lambda _checked=False, chosen=code: self.set_language(chosen),
            )
            self._language_group.addAction(action)
            self._language_menu.addAction(action)

    def set_language(self, code: str) -> None:
        """Persist the UI language choice into the config file; applied on the next launch."""
        if code == self._language:
            return
        self.used.emit("ui_language")
        self._language = code
        if self._rewrite_config(lambda existing: config_text_with_language(existing, code)):
            self.status.emit(_("Language saved. Restart Photo Tagger to apply it."))

    def _build_output_language_menu(self) -> None:
        """Add the Metadata Language submenu: one click sets the generated-metadata language."""
        menu = self._output_language_menu = self.addMenu(_("Metadata Language"))
        menu.setToolTipsVisible(True)
        menu.menuAction().setToolTip(
            tooltip(
                "Language of the generated titles, descriptions, and keywords. The language of "
                "the app itself is set under Language.",
            ),
        )
        self._output_language_group = QActionGroup(self)
        self._output_language_actions: dict[str, QAction] = {}
        # The trailing rows go in first so the language entries can insert above the separator
        # (a language chosen later via Other... joins the same list).
        self._output_language_separator = menu.addSeparator()
        other = menu.addAction(_("Other..."), self._choose_other_output_language)
        other.setToolTip(
            tooltip(
                "Any language name the model understands works; it is sent to the model as is. "
                "Saved to the config file, which the CLI's --output-language default also reads.",
            ),
        )
        # English (the default) first, then the suggestions sorted by their translated label.
        rest = sorted(
            (name for name in OUTPUT_LANGUAGE_SUGGESTIONS if name != DEFAULT_OUTPUT_LANGUAGE),
            key=lambda name: _(name).casefold(),
        )
        for name in (DEFAULT_OUTPUT_LANGUAGE, *rest):
            self._add_output_language_action(name)
        self._check_output_language_action(self._output_language)

    def _add_output_language_action(self, name: str) -> QAction:
        """
        Insert a checkable entry for *name* above the Other...

        row and return it.
        """
        # The label is translated for display; *name* (the English form) is what the prompt,
        # config file, and cache namespace carry. A custom name passes through _() unchanged.
        action = QAction(_(name), self)
        action.setCheckable(True)
        action.triggered.connect(
            lambda _checked=False, chosen=name: self.set_output_language(chosen),
        )
        self._output_language_group.addAction(action)
        self._output_language_menu.insertAction(self._output_language_separator, action)
        self._output_language_actions[name] = action
        return action

    def _check_output_language_action(self, name: str) -> None:
        """Check *name*'s menu entry, first creating one for a language not in the list."""
        action = self._output_language_actions.get(name)
        if action is None:
            action = self._add_output_language_action(name)
        action.setChecked(True)

    def _choose_other_output_language(self) -> None:
        """Ask for a free-form language name (Other...) and apply it."""
        language, ok = QInputDialog.getText(
            self.parentWidget(),
            _("Metadata Language"),
            _("Language for the generated titles, descriptions, and keywords:"),
            text=self._output_language,
        )
        if ok:
            self.set_output_language(language)

    def set_output_language(self, language: str) -> None:
        """Persist the metadata language into the config file; used from the next generation on."""
        normalized = language.strip() or DEFAULT_OUTPUT_LANGUAGE
        if normalized == self._output_language:
            # Still sync the menu: Other... may have re-entered the current language, and the
            # clicked action must not end up unchecked.
            self._check_output_language_action(normalized)
            return
        self.used.emit("output_language")
        self._output_language = normalized
        self._check_output_language_action(normalized)
        if not self._rewrite_config(
            lambda existing: config_text_with_output_language(existing, normalized),
        ):
            return
        self.output_language_changed.emit(normalized)
        self.status.emit(
            _("Metadata language set to {language} for the next generation.").format(
                language=normalized,
            ),
        )

    def _rewrite_config(self, rewrite: Callable[[str], str]) -> bool:
        """Apply *rewrite* to the config file's text; report whether the file was written."""
        target = config_target()
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rewrite(existing), encoding="utf-8")
        except (OSError, ConfigFileError) as exc:
            # ConfigFileError: the file on disk is not TOML we can rewrite in place. Reporting it
            # is the whole point of the dialog; letting it out of a Qt slot crashed the window.
            QMessageBox.warning(self.parentWidget(), _(CONFIG_SAVE_ERROR_TITLE), str(exc))
            return False
        return True

    def consent(self) -> telemetry.Consent:
        """Resolve whether this window may send, from its toggle and the environment."""
        return telemetry.resolve_consent(
            enabled=self._telemetry_enabled,
            opted_in=self._telemetry_opted_in,
        )

    def telemetry_choice(self) -> tuple[bool, bool]:
        """Return whether telemetry is on, and whether it was turned on by hand (an opt-in)."""
        return self._telemetry_enabled, self._telemetry_opted_in

    def turn_telemetry_off(self) -> None:
        """Switch telemetry off as if the toggle were unchecked, which persists the choice."""
        self._telemetry_action.setChecked(False)

    def _refresh_telemetry_action(self) -> None:
        """Check the toggle only when beacons go out; gray it out under an environment opt-out."""
        consent = self.consent()
        with QSignalBlocker(self._telemetry_action):
            self._telemetry_action.setChecked(consent.send)
        self._telemetry_action.setEnabled(consent.reason != telemetry.CONSENT_ENV_OPT_OUT)
        text = tooltip(
            "Anonymous usage stats (the settings and features you use, model, batch size, timing, "
            "OS, CPU/GPU, RAM, battery, screen) and crash reports (error type and stack trace, "
            "code locations only). No photos, text you type, error messages, or personal data.",
        )
        if note := telemetry_toggle_note(consent.reason, do_not_track=telemetry.do_not_track()):
            text = f"{text}\n\n{wrap_tooltip(note)}"
        self._telemetry_action.setToolTip(text)

    def _on_telemetry_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Persist the choice; checking the box is an opt-in that overrules DO_NOT_TRACK."""
        self._telemetry_enabled = enabled
        self._telemetry_opted_in = enabled
        telemetry.write_gui_pref(enabled=enabled)
        self._refresh_telemetry_action()
        self.status.emit(
            _("Anonymous telemetry on.") if enabled else _("Anonymous telemetry off."),
        )

    @property
    def output_language(self) -> str:
        """Return the language the model writes titles, descriptions, and keywords in."""
        return self._output_language

    @property
    def repair_language(self) -> bool:
        """Report whether a run asks again when an answer mixes writing systems."""
        return self._repair_language

    @property
    def records_undo(self) -> bool:
        """Report whether saves are recorded, so Tools > Undo Writes can put them back."""
        return self._undo_log_action.isChecked()

    @property
    def uses_cache(self) -> bool:
        """Report whether generation reuses earlier results for unchanged photos."""
        return self._cache_action.isChecked()

    def active_cache_file(self) -> Path | None:
        """Return the cache file generation should use, or None when caching is toggled off."""
        return self._cache_file if self._cache_action.isChecked() else None

    def notify_run_finished(self, message: str) -> None:
        """
        Raise the desktop alert for a run that just ended, when the user asked for one.

        *message* is the status line the run wrote, so the notification and the window agree on what
        happened. Never blocking: the alert spawns a helper process, and the window has to stay
        responsive while it does.
        """
        notify(
            APP_NAME,
            message,
            enabled=self._notify_action.isChecked(),
            sound=self._notify_sound_action.isChecked(),
        )

    def config_values(self) -> dict[str, bool]:
        """Return the toggles as the config file stores them, keyed like GuiConfigValues."""
        return {
            "telemetry_enabled": self._telemetry_enabled,
            "undo_log": self._undo_log_action.isChecked(),
            "notify": self._notify_action.isChecked(),
            "notify_sound": self._notify_sound_action.isChecked(),
            "repair_language": self._repair_language,
        }
