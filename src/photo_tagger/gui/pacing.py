# mypy: ignore-errors
# static analysis: ignore
# The two lines above keep zuban and pycroscope out; see the note in gui/__init__.py.
"""The Settings > Pacing dialog: how hard a run may push the machine."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from photo_tagger.gui.widgets import close_row, hint_label
from photo_tagger.gui_state import pacing_summary, tooltip
from photo_tagger.i18n import _
from photo_tagger.pacing import MAX_BATTERY_FLOOR, SECONDS_PER_MINUTE, PacingPlan


class PacingDialog(QDialog):
    """
    Decide how much of the time a run is allowed to work, and when it rests.

    The window rests exactly as a CLI run does, so an overnight batch here is as gentle as one
    there. Every edit is read straight into :attr:`plan`; a run already in flight keeps the plan it
    started with.
    """

    def __init__(self, parent: QWidget, plan: PacingPlan) -> None:
        """Build the dialog on *plan*, the one the config file's [pacing] table describes."""
        super().__init__(parent)
        self._plan = plan
        self.setWindowTitle(_("Pacing"))
        self.setMinimumWidth(620)
        box = QVBoxLayout(self)
        # Broken by hand: a QLabel sizes itself to its longest line, so one long sentence would
        # open the dialog as wide as the sentence is.
        blurb = QLabel(
            _("A model at full tilt can draw more power than the charger supplies.")
            + "\n"
            + _("Resting now and then lets the battery catch up and the fans wind down."),
        )
        blurb.setWordWrap(True)
        box.addWidget(blurb)
        box.addLayout(self._build_form(plan))

        # What the settings above add up to, in the terms a person actually asked about: how long
        # the run stops for, and how often. A percentage does not answer that on its own.
        self._pacing_preview = QLabel()
        self._pacing_preview.setWordWrap(True)
        self._pacing_preview.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse,
        )
        box.addWidget(self._pacing_preview)
        box.addWidget(
            hint_label(
                _(
                    "A rest falls between photos, never in the middle of one. Kept for this "
                    "session; Settings > Save Settings as Defaults writes it to the config file, "
                    "which CLI runs read too.",
                ),
            ),
        )
        box.addLayout(close_row(self))
        self._refresh()

    @property
    def plan(self) -> PacingPlan:
        """Return the plan the next run follows."""
        return self._plan

    def _build_form(self, plan: PacingPlan) -> QFormLayout:
        """Build the five rows the dialog edits, seeded from *plan*."""
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._duty_spin = QSpinBox()
        self._duty_spin.setRange(1, 100)
        self._duty_spin.setSuffix(" %")
        self._duty_spin.setValue(plan.duty_percent)
        self._duty_spin.setToolTip(
            tooltip(
                "Share of the time a run may work. 100 works flat out. 70 rests about three "
                "minutes of every ten, which is usually enough to stop a laptop losing charge.",
            ),
        )
        form.addRow(_("Work only"), self._duty_spin)

        self._battery_floor_spin = QSpinBox()
        # Up to one short of full, so a resume level always fits above it. A floor in the nineties
        # is a poor idea rather than an impossible one (see CAUTION_BATTERY_FLOOR), and the summary
        # says so instead of the box quietly refusing the number that was typed.
        self._battery_floor_spin.setRange(0, MAX_BATTERY_FLOOR)
        self._battery_floor_spin.setSuffix(" %")
        self._battery_floor_spin.setValue(plan.battery_floor)
        self._battery_floor_spin.setToolTip(
            tooltip(
                "Rest until the charge comes back whenever the battery falls this low. 0 never "
                "looks at the battery. Only rests while a charger is plugged in, since on "
                "battery power a rest recharges nothing.",
            ),
        )
        form.addRow(_("Rest below"), self._battery_floor_spin)

        self._battery_resume_spin = QSpinBox()
        self._battery_resume_spin.setRange(0, 100)
        self._battery_resume_spin.setSuffix(" %")
        self._battery_resume_spin.setValue(plan.battery_resume)
        self._battery_resume_spin.setToolTip(
            tooltip(
                "Charge that ends the rest. It has to sit above the floor, or the run would rest "
                "again as soon as it resumed, so raising the floor pushes this up with it.",
            ),
        )
        form.addRow(_("Resume at"), self._battery_resume_spin)

        self._work_minutes_spin = QDoubleSpinBox()
        self._work_minutes_spin.setRange(0.1, 240.0)
        self._work_minutes_spin.setDecimals(1)
        self._work_minutes_spin.setSuffix(_(" min"))
        self._work_minutes_spin.setValue(plan.work_seconds / SECONDS_PER_MINUTE)
        self._work_minutes_spin.setToolTip(
            tooltip(
                "How long a run works between rests. Resting after every photo would be too "
                "short to charge anything and gives the model server time to unload the model.",
            ),
        )
        form.addRow(_("Work for"), self._work_minutes_spin)

        self._max_pause_spin = QDoubleSpinBox()
        self._max_pause_spin.setRange(0.1, 240.0)
        self._max_pause_spin.setDecimals(1)
        self._max_pause_spin.setSuffix(_(" min"))
        self._max_pause_spin.setValue(plan.max_pause_seconds / SECONDS_PER_MINUTE)
        self._max_pause_spin.setToolTip(
            tooltip(
                "Longest a single rest may last. Bounds the wait for a battery that is charging "
                "slowly, and keeps a rest under the idle timeout that unloads a model.",
            ),
        )
        form.addRow(_("Rest at most"), self._max_pause_spin)

        for spin in (
            self._duty_spin,
            self._battery_floor_spin,
            self._battery_resume_spin,
            self._work_minutes_spin,
            self._max_pause_spin,
        ):
            spin.valueChanged.connect(self._refresh)
        return form

    def _hold_resume_above_the_floor(self) -> None:
        """
        Keep the resume level above the floor, so the pair can only ever mean something.

        Resuming at or below the charge that started the rest would rest again a photo later, and
        the plan quietly substitutes a level above the floor when asked for one that low. Better
        that the dialog never asks: the floor becomes the spin box's lower bound, and a floor raised
        past the resume level carries it up to the same margin the CLI would have picked.
        """
        floor = self._battery_floor_spin.value()
        spin = self._battery_resume_spin
        with QSignalBlocker(spin):
            if floor <= 0:
                spin.setMinimum(0)
                spin.setValue(0)
                return
            too_low = spin.value() <= floor
            spin.setMinimum(floor + 1)
            if too_low:
                spin.setValue(PacingPlan(battery_floor=floor).resume_percent)

    def _refresh(self) -> None:
        """
        Re-read the dialog into the plan the next run follows.

        A field is greyed out while nothing would read it: the two windows shape a rest, so they
        wait for something to ask for one, and the resume level waits for a floor to resume from.
        """
        self._hold_resume_above_the_floor()
        self._plan = PacingPlan(
            duty_percent=self._duty_spin.value(),
            battery_floor=self._battery_floor_spin.value(),
            battery_resume=self._battery_resume_spin.value(),
            work_seconds=self._work_minutes_spin.value() * SECONDS_PER_MINUTE,
            max_pause_seconds=self._max_pause_spin.value() * SECONDS_PER_MINUTE,
        )
        rests = self._plan.rests
        self._battery_resume_spin.setEnabled(self._plan.battery_floor > 0)
        for spin in (self._work_minutes_spin, self._max_pause_spin):
            spin.setEnabled(rests)
        self._pacing_preview.setText(pacing_summary(self._plan))
