import json

import numpy as np
import pytest

from adm.io import write_json


def test_arrays_become_row_major_lists(tmp_path):
    path = write_json({"U": np.array([[1.0, 2.0], [3.0, 4.0]]), "n": np.int64(2)}, tmp_path / "a" / "b.json")
    assert json.loads(path.read_text()) == {"U": [[1.0, 2.0], [3.0, 4.0]], "n": 2}


def test_nan_raises_and_writes_nothing(tmp_path):
    path = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        write_json({"x": np.array([np.nan])}, path)
    assert not path.exists()


def test_unsupported_type_raises(tmp_path):
    with pytest.raises(TypeError):
        write_json({"x": object()}, tmp_path / "x.json")
