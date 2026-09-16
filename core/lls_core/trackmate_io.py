"""
trackmate_io.py

Load ROI trajectories exported from TrackMate ("Export tracks to XML file",
the lightweight track-only export, not the full TrackMate XML project file)
and convert them into the data format napari's ``Tracks`` layer expects.

TrackMate's "tracks-only" XML looks like this::

    <Tracks nTracks="9" spaceUnits="micron" frameInterval="1.0" timeUnits="sec" ...>
      <particle nSpots="41">
        <detection t="0" x="47.84" y="0.0" z="0.0" />
        <detection t="1" x="48.71" y="0.0" z="0.0" />
        ...
      </particle>
      ...
    </Tracks>

napari's Tracks layer wants an (N, D+2) array where each row is
``[track_id, t, (z), y, x]`` and rows are sorted by increasing track_id then
time (see https://napari.org/stable/howtos/layers/tracks.html). Note the
axis order flip: TrackMate stores spots as (x, y, z) but napari's array
uses (z, y, x), matching image-axis order.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def load_trackmate_xml(xml_path: PathLike) -> Dict[str, dict]:
    """
    Parse a TrackMate "tracks-only" XML export into a plain dict:

        tracks = {'0': {'nSpots': 41, 'trackData': np.ndarray of shape (41, 4)}, ...}

    where each row of ``trackData`` is ``[t, x, y, z]`` (TrackMate's native
    order), matching the structure produced by ``importTrackMateTracks.m``.

    Parameters
    ----------
    xml_path : str or Path
        Path to the XML file produced by TrackMate's
        "Export tracks to XML file" action.

    Returns
    -------
    dict
        Mapping from track index (as a string) to a dict with keys
        ``'nSpots'`` and ``'trackData'``.

    Raises
    ------
    FileNotFoundError
        If ``xml_path`` does not exist.
    ValueError
        If the file cannot be parsed as TrackMate track XML.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"TrackMate XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse '{xml_path}' as XML: {exc}") from exc

    root = tree.getroot()
    if root.tag != "Tracks":
        raise ValueError(
            f"'{xml_path}' does not look like a TrackMate tracks export "
            f"(expected root tag <Tracks>, found <{root.tag}>)."
        )

    particles = root.findall("particle")
    n_tracks = int(root.attrib.get("nTracks", len(particles)))
    if n_tracks != len(particles):
        # Don't hard-fail on a mismatched count attribute - just trust the
        # actual number of <particle> elements found.
        n_tracks = len(particles)

    tracks: Dict[str, dict] = {}
    for i, particle in enumerate(particles):
        detections = particle.findall("detection")
        n_spots = int(particle.attrib.get("nSpots", len(detections)))

        track_data = np.empty((len(detections), 4), dtype=float)
        for j, detection in enumerate(detections):
            track_data[j] = (
                float(detection.attrib["t"]),
                float(detection.attrib["x"]),
                float(detection.attrib["y"]),
                float(detection.attrib["z"]),
            )
        # TrackMate does not guarantee detections are written in time order,
        # so sort defensively on the time column.
        track_data = track_data[np.argsort(track_data[:, 0])]

        tracks[str(i)] = {"nSpots": n_spots, "trackData": track_data}

    return tracks


def trackmate_tracks_to_napari(
    tracks: Dict[str, dict],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert the dict produced by :func:`load_trackmate_xml` into the
    ``(data, properties)`` pair expected by ``napari.Viewer.add_tracks``.

    Parameters
    ----------
    tracks : dict
        Output of :func:`load_trackmate_xml`.

    Returns
    -------
    data : np.ndarray, shape (N, 5)
        Columns are ``[track_id, t, z, y, x]``, sorted by increasing
        track_id then time, as required by napari's Tracks layer.
    properties : dict
        A ``{'nSpots': np.ndarray}`` features table (one value per vertex,
        constant within a track) that can be passed as ``features=`` /
        ``properties=`` to ``add_tracks`` so it's available for
        colour-by-feature in the GUI.
    """
    if not tracks:
        raise ValueError("No tracks to convert - the tracks dict is empty.")

    rows = []
    n_spots_col = []
    for track_id_str in sorted(tracks, key=int):
        track_id = int(track_id_str)
        track_data = tracks[track_id_str]["trackData"]  # (n, 4) = t, x, y, z
        n_spots = tracks[track_id_str]["nSpots"]
        for t, x, y, z in track_data:
            # napari column order is track_id, t, z, y, x
            rows.append((track_id, t, z, y, x))
            n_spots_col.append(n_spots)

    data = np.asarray(rows, dtype=float)
    # Sort by track_id then t, per napari's requirement.
    order = np.lexsort((data[:, 1], data[:, 0]))
    data = data[order]
    properties = {"nSpots": np.asarray(n_spots_col, dtype=int)[order]}

    return data, properties


def trackmate_xml_to_napari_tracks(
    xml_path: PathLike,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Convenience wrapper: parse an XML file straight to napari-ready data."""
    tracks = load_trackmate_xml(xml_path)
    return trackmate_tracks_to_napari(tracks)
