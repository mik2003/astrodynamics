import os
from typing import Literal

import numpy as np
import pytest
from numpy.typing import ArrayLike

from project.utils import Dir
from project.utils.cache import Memmap, simstate_view_from_state_view, write_simstate

FILENAME_MM_TEST = Dir.test / "test__1__2.simstate"


def make_test_file() -> None:
    data = simstate_view_from_state_view(
        np.array(
            [
                np.arange(12, dtype=np.float64),
                np.arange(12, dtype=np.float64) + 12,
                np.arange(12, dtype=np.float64) + 24,
            ]
        ),
        2,
    )
    write_simstate(FILENAME_MM_TEST, data)


@pytest.mark.parametrize(
    "rv,step,body,expected",
    [
        ("r", 0, 0, [[[0, 1, 2]]]),
        ("v", 0, 0, [[[6, 7, 8]]]),
        ("r", 0, 1, [[[3, 4, 5]]]),
        ("v", 0, 1, [[[9, 10, 11]]]),
        ("r", 1, 0, [[[12, 13, 14]]]),
        ("v", 1, 0, [[[18, 19, 20]]]),
        ("r", 1, 1, [[[15, 16, 17]]]),
        ("v", 1, 1, [[[21, 22, 23]]]),
        ("r", 2, 0, [[[24, 25, 26]]]),
        ("v", 2, 0, [[[30, 31, 32]]]),
        ("r", 2, 1, [[[27, 28, 29]]]),
        ("v", 2, 1, [[[33, 34, 35]]]),
        ("r", slice(0, 2), 0, [[[0, 1, 2]], [[12, 13, 14]]]),
        ("r", 0, slice(0, 2), [[[0, 1, 2], [3, 4, 5]]]),
        # Single step, all bodies
        ("r", 1, slice(None), [[[12, 13, 14], [15, 16, 17]]]),
        ("v", 2, slice(None), [[[30, 31, 32], [33, 34, 35]]]),
        # All steps, single body
        ("r", slice(None), 0, [[[0, 1, 2]], [[12, 13, 14]], [[24, 25, 26]]]),
        ("v", slice(None), 1, [[[9, 10, 11]], [[21, 22, 23]], [[33, 34, 35]]]),
        # All steps, all bodies
        (
            "r",
            slice(None),
            slice(None),
            [
                [[0, 1, 2], [3, 4, 5]],
                [[12, 13, 14], [15, 16, 17]],
                [[24, 25, 26], [27, 28, 29]],
            ],
        ),
        (
            "v",
            slice(None),
            slice(None),
            [
                [[6, 7, 8], [9, 10, 11]],
                [[18, 19, 20], [21, 22, 23]],
                [[30, 31, 32], [33, 34, 35]],
            ],
        ),
        # Step slice with stride
        ("r", slice(0, 3, 2), 1, [[[3, 4, 5]], [[27, 28, 29]]]),
        ("v", slice(0, 3, 2), 0, [[[6, 7, 8]], [[30, 31, 32]]]),
    ],
)
def test_memmap(
    rv: Literal["r", "v"],
    step: int | slice,
    body: int | slice,
    expected: ArrayLike,
) -> None:
    if not os.path.exists(FILENAME_MM_TEST):
        make_test_file()
    mm = Memmap(FILENAME_MM_TEST)
    actual = getattr(mm, rv)[step, body]
    np.testing.assert_allclose(actual, np.array(expected))
