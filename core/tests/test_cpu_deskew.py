#filename and function name should start with "test_" when using pytest
import numpy as np
import pytest
from lls_core import DeskewDirection, DeskewEngine
from lls_core.affine import determine_translation_and_bounding_box
from lls_core.cpu_deskew import cpu_deskew
from lls_core.cropping import Roi, track_to_rois
from lls_core.llsz_core import crop_volume_deskew, objective_crop_transforms
from lls_core.models.crop import CropParams
from lls_core.utils import ShapeOnly
from lls_core.models.lattice_data import LatticeData
from pydantic import ValidationError
from xarray import DataArray
import tempfile
from tests.utils import requires_real_gpu


def _feature_volume() -> np.ndarray:
    raw = np.zeros((10, 12, 8), dtype=np.float32)
    raw[5, 6, 4] = 100.0
    raw[2, 3, 1] = 40.0
    return raw


def _deskew(raw: np.ndarray, engine: str, skew: DeskewDirection, tmpdir: str) -> np.ndarray:
    lattice = LatticeData(
        input_image=DataArray(raw, dims=["Z", "Y", "X"]),
        # Isotropic XY pixel size: CPU and GPU engines agree on the output geometry
        # and (float32-precision) values whenever dx == dy. See cpu_deskew.py's
        # docstring for the anisotropic-XY caveat.
        physical_pixel_sizes=(0.3, 0.145, 0.145),
        angle=30.0,
        skew=skew,
        engine=engine,
        save_name="test",
        save_dir=tmpdir,
    )
    return np.asarray(next(iter(lattice.process().slices)).data)


@requires_real_gpu
@pytest.mark.parametrize("skew", [DeskewDirection.Y, DeskewDirection.X])
def test_cpu_engine_matches_gpu_engine(skew: DeskewDirection):
    raw = _feature_volume()
    with tempfile.TemporaryDirectory() as tmpdir:
        gpu = _deskew(raw, "GPU", skew, tmpdir)
        cpu = _deskew(raw, "CPU", skew, tmpdir)

    assert gpu.shape == cpu.shape
    np.testing.assert_allclose(gpu.astype(np.float64), cpu.astype(np.float64), atol=0.01)
    # The comparison must be non-trivial (both engines actually produced content)
    assert (gpu > 0).any()


def test_engine_accepts_string():
    raw = _feature_volume()
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(0.3, 0.145, 0.145),
            engine="CPU",
            save_name="test",
            save_dir=tmpdir,
        )
    assert lattice.engine == DeskewEngine.CPU


def _wide_volume(skew: DeskewDirection) -> np.ndarray:
    """Uniform background plus off-centre markers, wide along the shear axis, so an ROI
    placed anywhere along it contains structure (mirrors test_crop_placement.py)."""
    if skew == DeskewDirection.Y:
        raw = np.full((150, 400, 120), 30.0, np.float32)
        for z, a in [(30, 80), (75, 200), (120, 330)]:
            raw[z - 3:z + 3, a - 6:a + 6, 55:65] = 800.0
    else:
        raw = np.full((150, 120, 400), 30.0, np.float32)
        for z, a in [(30, 80), (75, 200), (120, 330)]:
            raw[z - 3:z + 3, 55:65, a - 6:a + 6] = 800.0
    return raw


@pytest.mark.parametrize("skew", [DeskewDirection.Y, DeskewDirection.X])
@pytest.mark.parametrize("angle", [30.0, 45.0])
@pytest.mark.parametrize("pos", ["low", "high"])  # off-centre, where mis-trimming is largest
def test_cpu_crop_matches_region_of_cpu_full_deskew(skew: DeskewDirection, angle: float, pos: str):
    """
    GPU-free counterpart of test_crop_placement.py: a CPU-engine ROI crop must equal the
    same region sliced out of the full CPU-deskewed volume.
    """
    raw = _wide_volume(skew)
    _, transform = objective_crop_transforms(raw.shape, angle, 1, 1, 1, skew)
    full_shape, _, _ = determine_translation_and_bounding_box(ShapeOnly(raw.shape), transform)
    full = cpu_deskew(raw, angle, 1, 1, 1, skew, tuple(full_shape))

    ax = 1 if skew == DeskewDirection.Y else 2   # sheared (wide) output axis
    other = 2 if skew == DeskewDirection.Y else 1
    size = 60
    frac = {"low": 0.2, "high": 0.8}[pos]
    start = int(np.clip(frac * full.shape[ax] - size / 2, 0, full.shape[ax] - size))
    stop = start + size
    b0 = full.shape[other] // 2 - size // 2
    b1 = b0 + size

    # ROI corners are [y, x]: a moving window on the shear axis, fixed on the other axis
    if skew == DeskewDirection.Y:
        roi = [[start, b0], [start, b1], [stop, b1], [stop, b0]]
        expected = full[:, start:stop, b0:b1]
    else:
        roi = [[b0, start], [b0, stop], [b1, stop], [b1, start]]
        expected = full[:, b0:b1, start:stop]
    assert expected.std() > 1.0  # otherwise the comparison is vacuous

    crop = np.asarray(crop_volume_deskew(
        original_volume=raw, roi_shape=[roi], angle_in_degrees=angle,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1,
        z_start=0, z_end=full.shape[0], skew_dir=skew, engine=DeskewEngine.CPU,
    ))

    assert crop.shape == expected.shape
    np.testing.assert_allclose(crop.astype(np.float64), expected.astype(np.float64), atol=1e-2)


@requires_real_gpu
@pytest.mark.parametrize("skew", [DeskewDirection.Y, DeskewDirection.X])
def test_cpu_crop_matches_gpu_crop(skew: DeskewDirection):
    raw = _wide_volume(skew)
    if skew == DeskewDirection.Y:
        roi = [[100, 30], [100, 90], [160, 90], [160, 30]]
    else:
        roi = [[30, 100], [30, 160], [90, 160], [90, 100]]
    common = dict(
        original_volume=raw, roi_shape=[roi], angle_in_degrees=30.0,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1,
        z_start=0, z_end=150, skew_dir=skew,
    )
    gpu = np.asarray(crop_volume_deskew(engine=DeskewEngine.GPU, **common)).astype(np.float64)
    cpu = np.asarray(crop_volume_deskew(engine=DeskewEngine.CPU, **common)).astype(np.float64)

    assert gpu.shape == cpu.shape
    np.testing.assert_allclose(gpu, cpu, atol=0.01)
    assert (gpu > 0).any()


def test_cpu_engine_allows_crop():
    """The CPU engine used to refuse a crop at validation time; it now runs one."""
    raw = _feature_volume()
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(0.3, 0.145, 0.145),
            angle=30.0,
            engine=DeskewEngine.CPU,
            crop=CropParams(roi_list=[[[0, 0], [0, 4], [4, 4], [4, 0]]]),
            save_name="test",
            save_dir=tmpdir,
        )
        slices = list(lattice.process().slices)

    assert lattice.engine == DeskewEngine.CPU
    assert len(slices) == 1
    assert slices[0].roi_index == 0
    data = np.asarray(slices[0].data)
    assert data.ndim == 3
    assert data.shape[1] == 4  # the ROI's extent along the skew (Y) axis


def test_cpu_engine_runs_a_tracked_crop():
    """A crop that follows a track (one window per timepoint) works on the CPU engine."""
    times = 3
    raw = np.full((times, 20, 80, 80), 10, dtype=np.uint16)
    # A TrackMate-driven crop carries a zero ROI as a placeholder; the real per-timepoint
    # windows live in `roi_by_time`, keyed by track then timepoint (microns until validated).
    track = np.asarray([[t, 10.0 + t, 10.0 + t, 0.0] for t in range(times)], dtype=float)
    placeholder = Roi((0, 0), (0, 0), (0, 0), (0, 0))
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["T", "Z", "Y", "X"]),
            physical_pixel_sizes=(1.0, 0.5, 0.5),
            engine=DeskewEngine.CPU,
            save_name="t",
            save_dir=tmpdir,
            save_type="tiff",
            crop=CropParams(
                roi_by_time={0: track_to_rois(track, window_size=10.0)},
                roi_list=[placeholder],
                roi_subset=[0],
                z_range=(0, 5),
            ),
        )
        slices = list(lattice.process().slices)

    assert [s.time for s in slices] == [0, 1, 2]
    for s in slices:
        data = np.asarray(s.data)
        assert data.ndim == 3
        assert data.max() > 0  # the window sat on the (uniform, non-zero) image


def test_cpu_engine_rejects_shear_only():
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="CPU deskew engine currently only supports"):
            LatticeData(
                input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                physical_pixel_sizes=(1, 1, 1),
                engine=DeskewEngine.CPU,
                coverslip_rotation=False,
                save_name="test",
                save_dir=tmpdir,
            )
