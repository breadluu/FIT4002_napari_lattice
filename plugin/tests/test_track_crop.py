"""
Headless tests for the plugin side of track-driven cropping. The methods are
called on lightweight stand-ins for `self`, so no napari viewer is needed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from lls_core.cropping import RoiUnits
from lls_core.models.crop import CropParams
from napari_lattice.dock_widget import LLSZWidget
from napari_lattice.fields import TrackmateFields


def fake_dock(fixed, tracked):
    container = SimpleNamespace(
        cropping_fields=SimpleNamespace(_make_model=lambda: fixed),
        trackmate_fields=SimpleNamespace(_make_model=lambda: tracked),
    )
    return SimpleNamespace(LlszMenu=SimpleNamespace(WidgetContainer=container))


FIXED = object()
TRACKED = object()


@pytest.mark.parametrize(
    "fixed, tracked, expected",
    [(None, None, None), (FIXED, None, FIXED), (None, TRACKED, TRACKED)],
)
def test_make_crop_model_picks_enabled_tab(fixed, tracked, expected):
    assert LLSZWidget._make_crop_model(fake_dock(fixed, tracked)) is expected


def test_make_crop_model_refuses_both():
    with pytest.raises(ValueError, match="can only use one"):
        LLSZWidget._make_crop_model(fake_dock(FIXED, TRACKED))


TRACK = np.array([[0, 10, 20, 0], [1, 11, 21, 0]], dtype=float)


def fake_trackmate(enabled=True, track=TRACK, window=4.0):
    return SimpleNamespace(
        fields_enabled=SimpleNamespace(value=enabled),
        window_size=SimpleNamespace(value=window),
        _get_selected_track_data=lambda: track,
    )


def test_disabled_returns_none():
    assert TrackmateFields._make_model(fake_trackmate(enabled=False)) is None


def test_disabled_does_not_require_a_track():
    # 'All' selected but tab not enabled -> just viewing tracks, no error
    assert TrackmateFields._make_model(fake_trackmate(enabled=False, track=None)) is None


def test_all_tracks_selected_is_rejected():
    with pytest.raises(ValueError, match="single track"):
        TrackmateFields._make_model(fake_trackmate(track=None))


def test_builds_micron_crop_from_selected_track():
    crop = TrackmateFields._make_model(fake_trackmate(window=4.0))
    assert isinstance(crop, CropParams)
    assert crop.roi_units == RoiUnits.Microns
    assert set(crop.roi_by_time) == {0, 1}
    ys = [y for y, _ in crop.roi_by_time[0]]
    xs = [x for _, x in crop.roi_by_time[0]]
    assert (min(ys), max(ys), min(xs), max(xs)) == (18, 22, 8, 12)
