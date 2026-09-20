from __future__ import annotations
from typing import TYPE_CHECKING, Dict, NamedTuple, Tuple, List

from strenum import StrEnum

if TYPE_CHECKING:
    from lls_core.types import PathLike
    from typing_extensions import Self
    from numpy.typing import NDArray

RoiCoord = Tuple[float, float]


class RoiUnits(StrEnum):
    """
    The units an ROI file's coordinates are in.

    ROI files carry no unit, and the two are off by a factor of 1/dy (~6.9x at a
    0.145 um pixel), so it has to be declared. `Auto` takes it from the file type,
    which is right for the two formats we write and read; override it for a CSV
    from elsewhere. `CropParams.roi_list` is always pixels - microns are converted
    on the way in.
    """
    Auto = "Auto"
    Pixels = "Pixels"
    Microns = "Microns"


def units_for_path(roi_path: PathLike) -> RoiUnits:
    """
    The units an ROI file is expected to be in, from its type.

    ImageJ writes pixels. A napari shapes CSV holds whatever that layer's data
    coordinates were; saved from the plugin's crop layer - which is unscaled while
    the image layer carries the pixel size - those are canvas microns.
    """
    from pathlib import Path
    from os import fspath

    if Path(fspath(roi_path)).suffix.lower() == ".csv":
        return RoiUnits.Microns
    return RoiUnits.Pixels

class Roi(NamedTuple):
    top_left: RoiCoord
    top_right: RoiCoord
    bottom_left: RoiCoord
    bottom_right: RoiCoord

    @classmethod
    def from_array(cls, array: NDArray) -> Self:
        import numpy as np
        return Roi(*np.reshape(array, (-1, 2)).tolist())

def read_roi_array(roi: PathLike) -> NDArray:
    from read_roi import read_roi_file
    from numpy import array
    return array(read_roi_file(str(roi)))

def read_napari_csv(roi_path: PathLike) -> List[Roi]:
    """
    Read a shapes layer saved by napari (File > Save Selected Layer, .csv).

    One row per vertex, grouped by the `index` column:

        index,shape-type,vertex-index,axis-0,axis-1
        0,polygon,0,100.0,100.0

    Non-rectangular shapes become their bounding rectangle, as for ImageJ ROIs. Only
    the last two axes are used, matching the plugin's own shape-to-ROI conversion, so
    3D shapes are accepted and their leading axes ignored.
    """
    import csv
    from collections import OrderedDict
    from os import fspath

    shapes: "OrderedDict[str, List[RoiCoord]]" = OrderedDict()
    with open(fspath(roi_path), newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        axes = [name for name in columns if name.startswith("axis-")]
        if "index" not in columns or len(axes) < 2:
            raise Exception(
                f"{roi_path} is not a napari shapes CSV: expected an 'index' column and "
                f"at least two 'axis-N' columns, found {columns}"
            )
        for row in reader:
            shapes.setdefault(row["index"], []).append(
                (float(row[axes[-2]]), float(row[axes[-1]]))
            )

    roi_list = []
    for vertices in shapes.values():
        top = min(y for y, _ in vertices)
        bottom = max(y for y, _ in vertices)
        left = min(x for _, x in vertices)
        right = max(x for _, x in vertices)
        roi_list.append(Roi((top, left), (top, right), (bottom, right), (bottom, left)))

    if not roi_list:
        raise Exception(f"No shapes found in {roi_path}")
    return roi_list


def read_rois(roi_path: PathLike) -> List[Roi]:
    """
    Read ROIs from an ImageJ .roi/.zip or a napari shapes .csv.

    Coordinates are returned as they are stored; see `RoiUnits` for why the caller
    must know whether they are pixels or microns.
    """
    from pathlib import Path
    from os import fspath

    if Path(fspath(roi_path)).suffix.lower() == ".csv":
        return read_napari_csv(roi_path)
    return read_imagej_roi(roi_path)


def scale_rois(rois: List[Roi], factor: float) -> List[Roi]:
    """Multiply every ROI coordinate by `factor`, e.g. to convert microns to pixels."""
    return [
        Roi(*[(y * factor, x * factor) for y, x in roi])
        for roi in rois
    ]


def track_to_rois(track_data: NDArray, window_size: float) -> Dict[int, Roi]:
    """
    Takes a track (one x,y per timepoint t, in microns) and turns each point into a 
    fixed size crop window centered on that point.

    CURRENTLY: Timepoints the track does not cover are absent from the result rather
    than interpolated. Z is also not used.
    """
    half = window_size / 2
    rois: Dict[int, Roi] = {}
    for t, x, y, _z in track_data:
        top, bottom = y - half, y + half
        left, right = x - half, x + half
        rois[int(t)] = Roi((top, left), (top, right), (bottom, right), (bottom, left))
    return rois


def clamp_rois_to_image(rois: Dict[int, Roi], height: float, width: float) -> Dict[int, Roi]:
    """
    Takes a set of per-timepoint crop windows and slides each one back inside the
    image bounds if it's hanging off an edge, without changing it's size. Otherwise,
    a moving crop that does this would be trimmed there, and frames that differ in
    size cannot be written as one image. A crop window larger than the image still 
    gets trimmed, but identically at every timepoint, so each frame still matches.
    """
    clamped: Dict[int, Roi] = {}
    for time, roi in rois.items():
        ys = [y for y, _ in roi]
        xs = [x for _, x in roi]
        top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)

        shift_y = -top if top < 0 else (height - bottom if bottom > height else 0.0)
        shift_x = -left if left < 0 else (width - right if right > width else 0.0)

        top, bottom = top + shift_y, bottom + shift_y
        left, right = left + shift_x, right + shift_x
        clamped[time] = Roi((top, left), (top, right), (bottom, right), (bottom, left))
    return clamped


def read_imagej_roi(roi_path: PathLike) -> List[Roi]:
    """Read an ImageJ ROI zip file so it loaded into napari shapes layer
        If non rectangular ROI, will convert into a rectangle based on extreme points
    Args:
        roi_zip_path (zip file): ImageJ ROI zip file

    Returns:
        list: List of ROIs
    """
    from pathlib import Path
    from os import fspath
    from read_roi import read_roi_file, read_roi_zip

    roi_path = Path(fspath(roi_path))

    # handle reading single roi or collection of rois in zip file
    if roi_path.suffix == ".zip":
        ij_roi = read_roi_zip(roi_path)
    elif roi_path.suffix == ".roi":
        ij_roi = read_roi_file(str(roi_path))
    else:
        raise Exception("ImageJ ROI file needs to be a zip/roi file")

    if ij_roi is None:
        raise Exception("Failed reading ROI file")

    # initialise list of rois
    roi_list = []

    # Read through each roi and create a list so that it matches the organisation of the shapes from napari shapes layer
    for value in ij_roi.values():
        if value['type'] in ('oval', 'rectangle'):
            width = int(value['width'])
            height = int(value['height'])
            left = int(value['left'])
            top = int(value['top'])
            roi = Roi((top, left), (top, left+width), (top+height, left+width), (top+height, left))
            roi_list.append(roi)
        elif value['type'] in ('polygon', 'freehand'):
            left = min(int(it) for it in value['x'])
            top = min(int(it) for it in value['y'])
            right = max(int(it) for it in value['x'])
            bottom = max(int(it) for it in value['y'])
            roi = Roi((top, left), (top, right), (bottom, right), (bottom, left))
            roi_list.append(roi)
        else:
            print(f"Cannot read ROI {value}. Recognised as type {value['type']}")

    return roi_list
