"""
Headless tests for the ROI Tracking tab (TrackmateFields): loading a track
file, the track ID dropdown, drawing the tracks layer and selecting one track.
Methods are called on stand-ins for `self`, so no napari viewer is needed.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from napari_lattice.fields import TrackmateFields

TRACKS = {
    "0": {"nSpots": 2, "trackData": np.array([[0, 10, 20, 1], [1, 11, 21, 1]], dtype=float)},
    "2": {"nSpots": 1, "trackData": np.array([[0, 30, 40, 2]], dtype=float)},
    "10": {"nSpots": 1, "trackData": np.array([[0, 50, 60, 3]], dtype=float)},
}

CSV = (
    "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R\n"
    ",,,(quality),(micron),(micron),(micron),(sec),,(micron)\n"
    "ID0,0,0,1.0,10.0,20.0,1.0,0.0,0,7.5\n"
    "ID1,1,5,1.0,30.0,40.0,2.0,0.0,0,7.5\n"
)


class FakeViewer:
    def __init__(self, layers=None):
        self.layers = dict(layers or {})
        self.added = []

    def add_tracks(self, data, properties=None, name=None, tail_width=None):
        self.added.append(dict(data=data, properties=properties, name=name, tail_width=tail_width))
        self.layers[name] = data


def fake_fields(tracks=None, selected="All", layer_name="", path=None):
    fake = SimpleNamespace(
        track_id=SimpleNamespace(value=selected, reset_choices=Mock()),
        layer_name=SimpleNamespace(value=layer_name),
        tracks_path=SimpleNamespace(value=path),
        _render_tracks_layer=Mock(),
    )
    if tracks is not None:
        fake._tracks = tracks
    return fake


def test_track_options_before_loading_are_just_all():
    assert TrackmateFields._get_track_id_options(fake_fields(), None) == ["All"]


def test_track_options_are_all_then_ids_in_numeric_order():
    options = TrackmateFields._get_track_id_options(fake_fields(TRACKS), None)
    assert options == ["All", "0", "2", "10"]


@pytest.mark.parametrize("selected", ["All", None, ""])
def test_no_single_track_selected_gives_none(selected):
    assert TrackmateFields._get_selected_track_data(fake_fields(TRACKS, selected)) is None


def test_no_file_loaded_gives_none():
    assert TrackmateFields._get_selected_track_data(fake_fields(selected="2")) is None


def test_selected_track_returns_its_track_data():
    data = TrackmateFields._get_selected_track_data(fake_fields(TRACKS, "2"))
    np.testing.assert_array_equal(data, TRACKS["2"]["trackData"])


def test_load_without_a_file_selected():
    with pytest.raises(ValueError, match="No file selected"):
        TrackmateFields.load_tracks(fake_fields(path=None))


def test_load_with_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        TrackmateFields.load_tracks(fake_fields(path=tmp_path / "nope.csv"))


def test_load_stores_tracks_resets_dropdown_and_draws(tmp_path):
    path = tmp_path / "spots.csv"
    path.write_text(CSV)
    fake = fake_fields(selected="3", path=path)

    TrackmateFields.load_tracks(fake)

    assert set(fake._tracks) == {"0", "5"}
    assert fake.track_id.value == "All"
    fake.track_id.reset_choices.assert_called_once()
    fake._render_tracks_layer.assert_called_once()


def test_changing_track_redraws_when_tracks_are_loaded():
    fake = fake_fields(TRACKS)
    TrackmateFields._on_track_id_changed(fake)
    fake._render_tracks_layer.assert_called_once()


def test_changing_track_does_nothing_before_loading():
    fake = fake_fields()
    TrackmateFields._on_track_id_changed(fake)
    fake._render_tracks_layer.assert_not_called()


@pytest.fixture
def viewer(monkeypatch):
    v = FakeViewer()
    monkeypatch.setattr("napari_lattice.utils.get_viewer", lambda: v)
    return v


def test_render_all_draws_every_track(viewer, tmp_path):
    TrackmateFields._render_tracks_layer(fake_fields(TRACKS, "All", path=tmp_path / "t.xml"))
    ids = set(viewer.added[0]["data"][:, 0])
    assert ids == {0, 2, 10}


def test_render_single_track_draws_only_that_track(viewer, tmp_path):
    TrackmateFields._render_tracks_layer(fake_fields(TRACKS, "2", path=tmp_path / "t.xml"))
    ids = set(viewer.added[0]["data"][:, 0])
    assert ids == {2}


def test_render_layer_name_defaults_to_file_stem(viewer, tmp_path):
    TrackmateFields._render_tracks_layer(fake_fields(TRACKS, path=tmp_path / "my_tracks.csv"))
    assert viewer.added[0]["name"] == "my_tracks"
    assert viewer.added[0]["tail_width"] == 2


def test_render_uses_trimmed_custom_layer_name(viewer, tmp_path):
    fake = fake_fields(TRACKS, layer_name="  cell A  ", path=tmp_path / "t.xml")
    TrackmateFields._render_tracks_layer(fake)
    assert viewer.added[0]["name"] == "cell A"


def test_render_replaces_an_existing_layer_with_the_same_name(monkeypatch, tmp_path):
    viewer = FakeViewer(layers={"t": "old layer"})
    monkeypatch.setattr("napari_lattice.utils.get_viewer", lambda: viewer)

    TrackmateFields._render_tracks_layer(fake_fields(TRACKS, path=tmp_path / "t.xml"))

    assert list(viewer.layers) == ["t"]
    assert isinstance(viewer.layers["t"], np.ndarray)  # the new tracks, not "old layer"
    assert len(viewer.added) == 1
