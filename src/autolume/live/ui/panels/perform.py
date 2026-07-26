"""Perform panel: the parameters played during a show.

Every row is the same row: a driver marker saying what drives it, then the
control. The model is one of them. It used to be a button and a label, drawn by
hand, which was already the odd one out and became a wrong one once a mapping
row could switch models from a controller: the parameter could be driven and
the row was the only one with nowhere to say so.

What the model is doing goes in the preview rather than here. A load takes
seconds and can fail, and the surface the performer is looking at while it
happens is the one showing the frames it is about to change.

The latent section carries two ways to navigate: the seed grid and a raw
vector, switched by `vector_mode`. Only one drives the frame at a time, so the
rows for the one not in play grey out rather than disappear, the same rule
every other conditional row in this panel already follows. The vector itself
is structured state, not a registry parameter (design.md), so it has no
address of its own and no driver marker: it reaches the control thread through
`/vector/set` and `/vector/randomize` instead.
"""

import dataclasses
import logging
import random
from pathlib import Path
from typing import Callable

import numpy as np
from imgui_bundle import imgui, portable_file_dialogs as pfd

from autolume.live.core.events import ControlEvent
from autolume.live.core.params import VECTOR_RANDOMIZE, VECTOR_SET, derive_mode
from autolume.live.errors import describe
from autolume.live.ui.controls import ControlBinder
from autolume.live.ui.theme import ERROR_COLOR

logger = logging.getLogger(__name__)

_BROWSE = "Browse"
_NO_MODEL = "No model loaded"
_VECTOR_FILTER = ["Vector files", "*.npy *.pt"]
_VECTOR_TENSOR_SUFFIXES = (".pt", ".pth")
_SEED_CEILING = 2**31 - 1


def button_width(label: str) -> float:
    """How wide a button reading `label` will be drawn.

    Measured rather than fixed, so the row keeps its proportions at every
    display scale and in whatever the theme sets for padding.
    """
    return imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 2.0


def load_vector_file(path: str) -> list[float]:
    """Read a `.npy` or a `.pt`/`.pth` file into a plain list of floats.

    Flattened and coerced here, so `/vector/set` always receives the same
    shape of data whichever format was on disk, the way the old app accepted
    either. `torch.load` is imported locally: nothing under `ui/` pays for it
    until a performer actually opens a vector file.
    """
    if Path(path).suffix.lower() in _VECTOR_TENSOR_SUFFIXES:
        import torch

        array = torch.load(path, map_location="cpu", weights_only=True).numpy()
    else:
        array = np.load(path)
    return [float(value) for value in np.asarray(array, dtype=np.float64).reshape(-1)]


def save_vector_file(path: str, vector: tuple[float, ...]) -> None:
    """Write `vector` to `path`, as a `.pt` tensor or an `.npy` array.

    `np.save` appends `.npy` itself when the name lacks it, which is also
    the format an unrecognized or missing suffix falls back to here.
    """
    if Path(path).suffix.lower() in _VECTOR_TENSOR_SUFFIXES:
        import torch

        torch.save(torch.tensor(vector, dtype=torch.float32), path)
    else:
        np.save(path, np.asarray(vector, dtype=np.float32))


class PerformPanel:
    def __init__(
        self, runtime, mapping_popup: Callable[[str], None] | None = None
    ) -> None:
        self._runtime = runtime
        self._binder = ControlBinder(runtime, mapping_popup)
        self._open_dialog: pfd.open_file | None = None
        self._open_vector_dialog: pfd.open_file | None = None
        self._save_vector_dialog: pfd.save_file | None = None
        self._vector_error: str | None = None

    def gui(self) -> None:
        self._model_row()
        self._latent_rows()
        self._noise_rows()
        self._render_rows()
        self._status_row()

    def _emit(self, address: str, value) -> None:
        self._runtime.submit(ControlEvent(address, value, source="ui"))

    def _model_row(self) -> None:
        """The model, as an ordinary row: driver marker, path, Browse.

        The field holds the path itself, so a path can be pasted or typed as
        well as picked, and it commits on Enter or on losing focus.

        Browse is drawn disabled exactly when the field is, which the binder
        reports: a source driving the parameter would overwrite whatever the
        dialog picked. The marker is drawn before either of them and stays
        clickable, so it is still the way back.
        """
        imgui.separator_text("Model")
        reserve = button_width(_BROWSE) + imgui.get_style().item_spacing.x
        live = self._binder.input_text(
            "pkl_path", "##model", hint=_NO_MODEL, reserve=reserve
        )
        imgui.same_line()
        if not live:
            imgui.begin_disabled()
        if imgui.button(_BROWSE):
            self._open_dialog = pfd.open_file(
                "Choose a model pickle", "", ["Model files", "*.pkl"]
            )
        if not live:
            imgui.end_disabled()
        # Drained outside the disabled block, so a dialog opened before a
        # source took the row over still delivers what it was opened for.
        self._take_dialog_result()

    def _take_dialog_result(self) -> None:
        if self._open_dialog is None or not self._open_dialog.ready():
            return
        result = self._open_dialog.result()
        self._open_dialog = None
        if result:
            self._emit("/model/path", str(result[0]))

    def _latent_rows(self) -> None:
        """Seed grid or raw vector, whichever `vector_mode` picks, plus motion.

        Both halves are drawn every frame, one of them greyed, so switching
        modes never reflows the panel. While a loop plays it takes the latent
        over entirely (design.md), which the seed rows already show on their
        own: `drives()` stands motion down for them during `loop_active`, so
        their marker reads as undriven exactly when the loop, not the seed
        walk, is the one moving the frame.
        """
        imgui.separator_text("Latent")
        self._binder.bool_radio("vector_mode", "Seed", "Vec")
        vector_mode = bool(self._binder.value("vector_mode"))
        # `derive_mode`, not a local re-derivation: `generator.py`'s
        # `render_frame` reads `latent_project` only in its `"vec"` branch and
        # the seed grid (`latent_x`/`latent_y`) only in its `"seed"` branch,
        # so both have to ask the exact question the generator dispatches on
        # or they can drift from it. `vector_mode` is swapped in from the
        # displayed value above rather than the raw snapshot, so toggling the
        # radio greys both in the same frame instead of one behind it.
        mode = derive_mode(
            dataclasses.replace(self._binder.state(), vector_mode=vector_mode)
        )
        self._binder.checkbox("latent_project", "Project", enabled=mode == "vec")
        # Was `enabled=not vector_mode`: correct outside a loop, but a loop
        # (either kind) takes the latent over entirely, per this method's own
        # docstring, and `_blended_w` (`generator.py`) is only ever called in
        # the `"seed"` branch, never while one plays. The marker already
        # showed this (`drives()` stands motion down during `loop_active`),
        # but the widget itself stayed live and editable regardless, the same
        # live-but-inert shape Project shipped with twice before this sweep.
        self._binder.drag_float("latent_x", "Latent x", enabled=mode == "seed")
        self._binder.drag_float("latent_y", "Latent y", enabled=mode == "seed")
        self._vector_row(vector_mode)
        self._binder.checkbox("anim_playing", "Animate")
        self._binder.slider_float("anim_speed_x", "Speed x")
        self._binder.slider_float("anim_speed_y", "Speed y")
        self._binder.slider_float("truncation_psi", "Truncation")

    def _vector_row(self, vector_mode: bool) -> None:
        """Randomize, Load and Save, greyed together outside vector mode.

        One block rather than a bound control: `latent_vec` is structured
        state with no registry address, so there is no gutter to draw beside
        it, the same reason the model row's Browse button carries none either.
        """
        if not vector_mode:
            imgui.begin_disabled()
        if imgui.button("Randomize"):
            self._emit(VECTOR_RANDOMIZE, random.randint(0, _SEED_CEILING))
        imgui.same_line()
        if imgui.button("Load"):
            self._open_vector_dialog = pfd.open_file(
                "Load a latent vector", "", _VECTOR_FILTER
            )
        imgui.same_line()
        if imgui.button("Save"):
            self._save_vector_dialog = pfd.save_file(
                "Save the latent vector", "vector.npy", _VECTOR_FILTER
            )
        if not vector_mode:
            imgui.end_disabled()
        # Drained outside the disabled block, matching the model dialog: a
        # dialog opened before the mode changed underneath it still delivers.
        self._take_vector_load()
        self._take_vector_save()
        if self._vector_error:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*ERROR_COLOR))
            imgui.text_wrapped(self._vector_error)
            imgui.pop_style_color()

    def _take_vector_load(self) -> None:
        if self._open_vector_dialog is None or not self._open_vector_dialog.ready():
            return
        result = self._open_vector_dialog.result()
        self._open_vector_dialog = None
        if not result:
            return
        self._vector_error = None
        try:
            values = load_vector_file(result[0])
        except Exception as exc:
            logger.exception("Could not load vector %s", result[0])
            self._vector_error = f"Could not load the vector. {describe(exc)}"
            return
        self._emit(VECTOR_SET, values)

    def _take_vector_save(self) -> None:
        if self._save_vector_dialog is None or not self._save_vector_dialog.ready():
            return
        path = self._save_vector_dialog.result()
        self._save_vector_dialog = None
        if not path:
            return
        self._vector_error = None
        try:
            save_vector_file(path, self._binder.state().latent_vec)
        except Exception as exc:
            logger.exception("Could not save vector %s", path)
            self._vector_error = f"Could not save the vector. {describe(exc)}"

    def _noise_rows(self) -> None:
        imgui.separator_text("Noise")
        self._binder.checkbox("noise_enabled", "Noise on")
        # What the box shows, not what the store holds, so the rows grey in the
        # same frame as the click that greys them.
        live = bool(self._binder.value("noise_enabled"))
        self._binder.slider_float("global_noise", "Amount", enabled=live)
        # A typed field, not a drag: neighbouring seeds are not a smooth
        # sweep, each is an unrelated noise pattern, so there is nothing
        # meaningful to watch happen while dragging between them.
        self._binder.input_int("noise_seed", "Seed", enabled=live)
        self._binder.checkbox("noise_anim", "Animate noise", enabled=live)

    def _render_rows(self) -> None:
        imgui.separator_text("Render")
        # A typed field: the frame limit is a setting the performer picks
        # once, not a value worth sweeping through and watching change.
        self._binder.input_int("fps_cap", "Frame limit")

    def _status_row(self) -> None:
        imgui.separator()
        imgui.text(f"Render {self._runtime.render_loop.fps():.1f} fps")
        if self._runtime.osc.port is not None:
            imgui.text(f"OSC port {self._runtime.osc.port}")
