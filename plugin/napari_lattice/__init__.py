from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari-lattice")
except PackageNotFoundError:
    # Editable/source checkout that is not installed, or a clone with no tags yet.
    __version__ = "0.0.0"
