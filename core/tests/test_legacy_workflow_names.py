"""
Workflow files written before the move from `pyclesperanto_prototype` to `pyclesperanto`
name their functions as `pyclesperanto_prototype.<name>`. Those must keep loading, on a
machine where the old package is not installed.
"""
import sys
from pathlib import Path

import pyclesperanto
import pytest

from lls_core.workflow import LEGACY_CLE_MODULE, workflow_from_path

TEMPLATE = """!!python/object:napari_workflows._workflow.Workflow
_tasks:
  blurred: !!python/tuple
  - !!python/name:{module}.{function} ''
  - deskewed_image
  - null
  - 1
  - 1
  - 1
"""


def _write(tmp_path: Path, module: str, function: str) -> Path:
    path = tmp_path / "flow.yml"
    path.write_text(TEMPLATE.format(module=module, function=function))
    return path


@pytest.fixture
def prototype_not_installed(monkeypatch):
    # A `None` entry makes `import pyclesperanto_prototype` raise ImportError, which is
    # what a clean install looks like.
    monkeypatch.setitem(sys.modules, LEGACY_CLE_MODULE, None)


def test_legacy_name_loads_from_pyclesperanto(tmp_path, prototype_not_installed):
    workflow = workflow_from_path(_write(tmp_path, LEGACY_CLE_MODULE, "gaussian_blur"))
    function = workflow._tasks["blurred"][0]
    assert function is pyclesperanto.gaussian_blur


def test_current_name_is_unaffected(tmp_path, prototype_not_installed):
    workflow = workflow_from_path(_write(tmp_path, "pyclesperanto", "gaussian_blur"))
    assert workflow._tasks["blurred"][0] is pyclesperanto.gaussian_blur


def test_alias_does_not_leak_out_of_the_load(tmp_path, prototype_not_installed):
    workflow_from_path(_write(tmp_path, LEGACY_CLE_MODULE, "gaussian_blur"))
    # Still the sentinel the fixture put there, not `pyclesperanto`
    assert sys.modules[LEGACY_CLE_MODULE] is None


def test_unknown_legacy_name_explains_itself(tmp_path, prototype_not_installed):
    path = _write(tmp_path, LEGACY_CLE_MODULE, "a_function_that_does_not_exist")
    with pytest.raises(ValueError, match="replaced with `pyclesperanto`"):
        workflow_from_path(path)
