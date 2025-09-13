import json
import os
from pathlib import Path
from typing import Any, Dict, List, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ValidationError, field_validator

from project.utilities import camel_to_snake

T = TypeVar("T", bound="Body")
A = npt.NDArray[np.float64]


class Vector3(np.ndarray):
    @staticmethod
    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=float).reshape(3, 1)
        obj = np.asarray(arr).view(cls)
        return obj


class Body(BaseModel):
    """Dataclass for body characterstics"""

    name: str | None = None
    mu: float | None = None

    r_0: Vector3 | None = None
    v_0: Vector3 | None = None

    r: List[Vector3] = []
    v: List[Vector3] = []
    a: List[Vector3] = []

    model_config = dict(arbitrary_types_allowed=True)

    @field_validator("r_0", "v_0", mode="before")
    def validate_vector(cls, v):  # pylint: disable=no-self-argument
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError("Vector must be a list of 3 elements")
        return Vector3(v)

    @classmethod
    def load(cls, body: Dict[str, Any]) -> "Body":
        """Load mission phase."""
        try:
            return cls(**body)
        except ValidationError as e:
            print(e.errors())
            raise

    def dump(self) -> Dict[str, Any]:
        """Dump body"""
        # Convert to dict first
        body_dict = self.model_dump()

        # Convert Vector3 arrays to lists
        for key in ["r_0", "v_0"]:
            if key in body_dict and body_dict[key] is not None:
                body_dict[key] = body_dict[key].tolist()

        # Convert lists of Vector3 to lists of lists
        for key in ["r", "v", "a"]:
            if key in body_dict and body_dict[key] is not None:
                body_dict[key] = [vec.tolist() for vec in body_dict[key]]

        return body_dict


class BodyList(list[Body]):
    """Class to list bodies"""

    def __init__(self, value) -> None:
        super().__init__(value)

        self._r_0: A | None = None
        self._v_0: A | None = None

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
        """Dump body list to file."""

        # Use lowercase class name as key
        key = camel_to_snake(self.__class__.__name__)

        save_data = []
        for body in self:
            body_data = body.dump()  # This already converts arrays to lists

            # Additional processing if needed for any remaining numpy arrays
            for body_key, value in body_data.items():
                if hasattr(value, "tolist"):  # Check if it's a numpy array
                    body_data[body_key] = value.tolist()
                elif (
                    isinstance(value, list)
                    and value
                    and hasattr(value[0], "tolist")
                ):
                    # Handle lists of numpy arrays
                    body_data[body_key] = [
                        v.tolist() if hasattr(v, "tolist") else v
                        for v in value
                    ]

            save_data.append(body_data)

        if not os.path.exists(file_path):
            data = {key: save_data}
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data[key] = save_data

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @property
    def r_0(self) -> A:
        if self._r_0 is None:
            self._r_0 = np.hstack([body.r_0 for body in self])
        return self._r_0

    @r_0.setter
    def r_0(self, value: A) -> None:
        self._r_0 = value

    @property
    def v_0(self) -> A:
        if self._v_0 is None:
            self._v_0 = np.hstack([body.v_0 for body in self])
        return self._v_0

    @v_0.setter
    def v_0(self, value: A) -> None:
        self._v_0 = value

    def a(self, r: A) -> A:
        """
        Compute accelerations for all bodies at given positions r.
        r is shaped (3, N), with each column representing a body's position.
        Returns accelerations in the same shape (3, N).
        """
        n_bodies = r.shape[1]
        a_mat = np.zeros_like(r)

        for i in range(n_bodies):
            r_i = r[:, i : i + 1]  # Keep as (3, 1) for broadcasting
            a_i = np.zeros((3, 1))

            for j in range(n_bodies):
                if i == j or self[j].mu is None:
                    continue

                r_j = r[:, j : j + 1]  # Keep as (3, 1) for broadcasting
                r_ij = r_j - r_i
                dist = np.linalg.norm(r_ij)

                if dist == 0:  # avoid division by zero
                    continue

                a_i += self[j].mu * r_ij / dist**3

            a_mat[:, i : i + 1] = a_i

        return a_mat

    def perform_step(self, h: float) -> None:
        # Store current positions and velocities for each body
        for i, body in enumerate(self):
            # Append current state to history lists
            body.r.append(Vector3(self.r_0[:, i]))
            body.v.append(Vector3(self.v_0[:, i]))

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
