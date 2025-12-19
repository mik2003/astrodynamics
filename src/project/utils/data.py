import json
import os
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypeVar

import numpy as np
import tomli_w
from pydantic import (
    BaseModel,
    Field,
    GetCoreSchemaHandler,
    ValidationError,
)
from pydantic_core import core_schema

from project.utils import A, camel_to_snake

T = TypeVar("T", bound="Body")


class Vector3(np.ndarray):
    def __new__(cls, data: Iterable[float]) -> "Vector3":
        arr = np.asarray(list(data), dtype=float)
        if arr.shape != (3,):
            raise ValueError("Vector3 must have exactly 3 elements")
        return arr.view(cls)

    def __array_finalize__(self, obj: object | None) -> None:
        # Required for ndarray subclasses
        if obj is None:
            return

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.list_schema(
                core_schema.float_schema(),
                min_length=3,
                max_length=3,
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.tolist(),
                return_schema=core_schema.list_schema(core_schema.float_schema()),
            ),
        )


Vector3Type = Annotated[Vector3, Field(description="3D Vector (3,)")]


class Body(BaseModel):
    """Dataclass for body characterstics"""

    name: str | None = None
    mu: float | None = None

    r_0: Vector3Type | None = None
    v_0: Vector3Type | None = None

    radius: float | None = None

    @classmethod
    def load(cls, body: Dict[str, Any]) -> "Body":
        """Load mission phase."""
        return cls(**body)

    def dump(self) -> Dict[str, Any]:
        """Dump body"""
        # Convert to dict first
        body_dict = self.model_dump()

        # Convert Vector3 arrays to lists
        for key in ["r_0", "v_0"]:
            if key in body_dict and body_dict[key] is not None:
                body_dict[key] = body_dict[key].tolist()

        return body_dict

    @classmethod
    def load2(cls, body: Dict[str, Any]) -> "Body":
        """Load mission phase."""
        try:
            return cls(**body)
        except ValidationError as e:
            print(e.errors())
            raise

    def dump2(self) -> Dict[str, Any]:
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

    def __init__(self, value: List[Body]) -> None:
        super().__init__(value)
        self.metadata: Dict[str, str] | None = None

        self._r_0: A | None = None
        self._v_0: A | None = None

    @staticmethod
    def load(file_path: Path) -> "BodyList":
        """Load mission profile from file."""
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
        bl_raw = data["body_list"]
        bl = BodyList([Body.load(body) for body in bl_raw])
        bl.metadata = data["metadata"]
        return bl

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
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            data[key] = save_data

        with open(file_path, "wb") as f:
            tomli_w.dump(data, f)

    @staticmethod
    def load2(file_path: Path) -> "BodyList":
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

    def dump2(self, file_path: Path) -> None:
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
