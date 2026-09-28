"""JSON helpers for machine-readable records.

Record writers may receive NumPy scalars or diagnostic values that are not
finite.  JSON's ``NaN`` and ``Infinity`` extensions are not accepted by a
strict parser, so those values are represented as ``null`` before encoding.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

_SCALAR_SEQUENCE_TYPES = (str, bytes, bytearray)


def normalize_json(value: Any) -> Any:
    """Return a JSON-compatible tree with non-finite numbers represented by ``None``.

    Mappings are copied with string keys, sequences become lists, and scalar
    values from NumPy (when present) are reduced through their Python scalar
    representation without importing NumPy.  Unknown values are left for
    ``json.dumps`` to reject rather than silently stringifying them.
    """
    if isinstance(value, Mapping):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, _SCALAR_SEQUENCE_TYPES):
        return [normalize_json(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None

    is_numpy = type(value).__module__.split(".", 1)[0] == "numpy"
    if is_numpy:
        # NumPy bool_ and zero-dimensional arrays do not register as the
        # standard numeric ABCs.  Their item/list conversion keeps this module
        # usable by the standard-library-only benchmark runner as well as the
        # reference path.
        item = getattr(value, "item", None)
        if callable(item):
            try:
                converted = item()
            except (TypeError, ValueError):
                pass
            else:
                if converted is not value:
                    return normalize_json(converted)

        # NumPy arrays are not registered as collections.abc.Sequence.
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                converted = tolist()
            except (TypeError, ValueError):
                pass
            else:
                if converted is not value:
                    return normalize_json(converted)

    return value


def dumps(value: Any, **kwargs: Any) -> str:
    """Serialize a record after normalization using standards-compliant JSON."""
    kwargs.pop("allow_nan", None)
    return json.dumps(normalize_json(value), allow_nan=False, **kwargs)
