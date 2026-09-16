"""
trackmate_widget.py

A small napari widget that lets a user pick a TrackMate "tracks-only" XML
file and adds it to the viewer as an interactive napari ``Tracks`` layer.
"""
from __future__ import annotations

from pathlib import Path

from magicgui import magic_factory

from core.lls_core.trackmate_io import trackmate_xml_to_napari_tracks

try:
    # Only needed for the type hint below; napari is a runtime dependency
    # of the plugin this widget lives in, so this import is safe.
    from napari.viewer import Viewer
except ImportError:  # pragma: no cover - keeps this file importable in tests
    Viewer = "napari.viewer.Viewer"


@magic_factory(
    call_button="Load TrackMate tracks",
    xml_path={
        "label": "TrackMate tracks XML",
        "filter": "TrackMate XML (*.xml)",
        "mode": "r",
    },
    layer_name={"label": "Layer name (optional)"},
)
def load_trackmate_tracks_widget(
    viewer: "Viewer",
    xml_path: Path = Path.home(),
    layer_name: str = "",
) -> None:
    """
    Load a TrackMate ROI-tracking XML export and add it to the viewer as a
    ``Tracks`` layer.

    Parameters
    ----------
    viewer : napari.Viewer
        Injected automatically by napari/magicgui - the currently active
        viewer instance.
    xml_path : Path
        Path to the XML file produced by TrackMate's
        "Export tracks to XML file" action.
    layer_name : str
        Optional name for the new layer. Defaults to the XML file's stem.
    """
    if not xml_path or not Path(xml_path).is_file():
        raise ValueError(f"Please choose a valid TrackMate XML file (got: {xml_path!r}).")

    data, properties = trackmate_xml_to_napari_tracks(xml_path)

    name = layer_name.strip() or Path(xml_path).stem
    viewer.add_tracks(
        data,
        properties=properties,
        name=name,
        tail_width=2,
    )