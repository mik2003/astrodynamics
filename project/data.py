import json
import os
from pathlib import Path
from typing import Any, Dict, List, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ValidationError, field_validator

from project.utilities import camel_to_snake

T = TypeVar("T", bound="Body")
Vector = npt.NDArray[np.float64]


class Vector3D(np.ndarray):
    @staticmethod
    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=float).reshape(3, 1)
        obj = np.asarray(arr).view(cls)
        return obj


class Body(BaseModel):
    """Dataclass for body characterstics"""

    name: str | None = None
    mu: float | None = None

    r_0: Vector3D | None = None
    v_0: Vector3D | None = None

    r: List[Vector3D] = []
    v: List[Vector3D] = []
    a: List[Vector3D] = []

    model_config = dict(arbitrary_types_allowed=True)

    @field_validator("r_0", "v_0", mode="before")
    def validate_vector(cls, v):  # pylint: disable=no-self-argument
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError("Vector must be a list of 3 elements")
        return Vector3D(v)

    @classmethod
    def load(cls, body: Dict[str, Any]) -> "Body":
        """Load mission phase."""
        try:
            return cls(**body)
        except ValidationError as e:
            print(e.errors())
            raise

    def dump(self) -> Dict[str, Any]:
        """Dump mission phase."""
        return self.model_dump()


class BodyList(list[Body]):
    """Class to list bodies"""

    def __init__(self, value) -> None:
        super().__init__(value)

        self._r_0: Vector | None = None
        self._v_0: Vector | None = None

    @staticmethod
    def load(file_path: Path) -> "BodyList":
        """Load mission profile from file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            bl_raw = data["body_list"]
            return BodyList([Body.load(body) for body in bl_raw])
        except ValidationError as e:
            print(e.errors())
            raise

    def dump(self, file_path: Path) -> None:
        """Dump mission profile to file."""

        # Use lowercase class name as key
        key = camel_to_snake(self.__class__.__name__)

        save_data = [body.dump() for body in self]

        if not os.path.exists(file_path):
            data = {key: save_data}
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data[key] = save_data

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @property
    def r_0(self) -> Vector:
        if self._r_0 is None:
            self._r_0 = np.hstack([body.r_0 for body in self])
        return self._r_0

    @r_0.setter
    def r_0(self, value: Vector) -> None:
        self._r_0 = value

    @property
    def v_0(self) -> Vector:
        if self._v_0 is None:
            self._v_0 = np.hstack([body.v_0 for body in self])
        return self._v_0

    @v_0.setter
    def v_0(self, value: Vector) -> None:
        self._v_0 = value

    def a(self, r: Vector) -> Vector:
        """
        Compute accelerations for all bodies at given positions r_vec.
        r_vec is shaped (3*N, 1), stacking all positions.
        Returns accelerations in the same stacked shape.
        """
        a_list = []

        for i, _ in enumerate(self):
            # extract position of body i from the stacked vector
            r_i = r[3 * i : 3 * (i + 1), :]
            a_i = np.zeros((3, 1))

            for j, body_j in enumerate(self):
                if i == j or body_j.mu is None:
                    continue

                r_j = r[3 * j : 3 * (j + 1), :]
                r_ij = r_j - r_i
                dist = np.linalg.norm(r_ij)

                if dist == 0:  # avoid division by zero
                    continue

                a_i += body_j.mu * r_ij / dist**3

            a_list.append(a_i)

        return np.concatenate(a_list, axis=0)

    def perform_step(self, h: float) -> None:
        for i, body in enumerate(self):
            body.r.append(self.r_0[3 * i : 3 * i + 3])
            body.v.append(self.v_0[3 * i : 3 * i + 3])

        k_r1 = self.v_0
        k_v1 = self.a(self.r_0)

        k_r2 = self.v_0 + k_v1 * h / 2
        k_v2 = self.a(self.r_0 + k_r1 * h / 2)

        k_r3 = self.v_0 + k_v2 * h / 2
        k_v3 = self.a(self.r_0 + k_r2 * h / 2)

        k_r4 = self.v_0 + k_v3 * h
        k_v4 = self.a(self.r_0 + k_r3 * h)

        self.r_0 = self.r_0 + h / 6 * (k_r1 + 2 * k_r2 + 2 * k_r3 + k_r4)
        self.v_0 = self.v_0 + h / 6 * (k_v1 + 2 * k_v2 + 2 * k_v3 + k_v4)
