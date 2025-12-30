import os
from typing import Literal

import numpy as np
import pytest
from numpy.typing import ArrayLike

from project.simulation import Memmap
from project.utils import Dir, append_positions_npy

FILENAME_MM_TEST = Dir.test / "test_1_1.bin"


def make_test_file() -> None:
    data = np.array([np.arange(12), np.arange(12) + 12, np.arange(12) + 24])
    append_positions_npy(FILENAME_MM_TEST, data)


@pytest.mark.parametrize(
    "step,body,value,expected",
    [
        (0, 0, "r", [[[0, 1, 2]]]),
        (0, 0, "v", [[[6, 7, 8]]]),
        (0, 1, "r", [[[3, 4, 5]]]),
        (0, 1, "v", [[[9, 10, 11]]]),
        (1, 0, "r", [[[12, 13, 14]]]),
        (1, 0, "v", [[[18, 19, 20]]]),
        (1, 1, "r", [[[15, 16, 17]]]),
        (1, 1, "v", [[[21, 22, 23]]]),
        (2, 0, "r", [[[24, 25, 26]]]),
        (2, 0, "v", [[[30, 31, 32]]]),
        (2, 1, "r", [[[27, 28, 29]]]),
        (2, 1, "v", [[[33, 34, 35]]]),
        (slice(0, 2), 0, "r", [[[0, 1, 2]], [[12, 13, 14]]]),
        (0, slice(0, 2), "r", [[[0, 1, 2], [3, 4, 5]]]),
    ],
)
def test_memmap(
    step: int | slice,
    body: int | slice,
    value: Literal["r", "v"],
    expected: ArrayLike,
) -> None:
    if not os.path.exists(FILENAME_MM_TEST):
        make_test_file()
    mm = Memmap(FILENAME_MM_TEST, 3, 2)
    actual = mm[step, body, value]
    assert actual.ndim == 3
    np.testing.assert_allclose(actual, np.array(expected))
