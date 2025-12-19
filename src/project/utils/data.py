import json
import os
from pathlib import Path
from typing import Any, Dict, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

from project.utils import A, camel_to_snake

T = TypeVar("T", bound="Body")


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

    radius: float | None = None

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

        return body_dict


class BodyList(list[Body]):
    """Class to list bodies"""

    def __init__(self, value) -> None:
        super().__init__(value)
        self.metadata: Dict[str, str] | None = None

        self._r_0: A | None = None
        self._v_0: A | None = None

    @staticmethod
    def load(file_path: Path) -> "BodyList":
        """Load mission profile from file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            bl_raw = data["body_list"]
            bl = BodyList([Body.load(body) for body in bl_raw])
            bl.metadata = data["metadata"]
            return bl
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
                elif isinstance(value, list) and value and hasattr(value[0], "tolist"):
                    # Handle lists of numpy arrays
                    body_data[body_key] = [
                        v.tolist() if hasattr(v, "tolist") else v for v in value
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
            self._r_0 = np.hstack([body.r_0 for body in self if body.r_0 is not None])
        return self._r_0

    @r_0.setter
    def r_0(self, value: A) -> None:
        self._r_0 = value

    @property
    def v_0(self) -> A:
        if self._v_0 is None:
            self._v_0 = np.hstack([body.v_0 for body in self if body.v_0 is not None])
        return self._v_0

    @v_0.setter
    def v_0(self, value: A) -> None:
        self._v_0 = value
