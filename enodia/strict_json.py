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

    Mappings are copied with string keys, sequences become lists, and NumPy
    arrays are converted with ``tolist`` before any scalar conversion so their
    shape is retained.  Finite extended-precision NumPy floats are represented
    as decimal strings instead of being narrowed to Python ``float``.  Unknown
    values are left for ``json.dumps`` to reject rather than silently
    stringifying them.
    """
    if isinstance(value, Mapping):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, _SCALAR_SEQUENCE_TYPES):
        return [normalize_json(item) for item in value]

    is_numpy = type(value).__module__.split(".", 1)[0] == "numpy"
    if is_numpy:
        # NumPy arrays are not registered as collections.abc.Sequence.  A
        # non-zero-dimensional array must be converted to nested lists before
        # considering its scalar conversion, otherwise a one-element array
        # would lose its shape.
        ndim = getattr(value, "ndim", None)
        if ndim is not None and ndim > 0:
            tolist = getattr(value, "tolist", None)
            if callable(tolist):
                try:
                    converted = tolist()
                except (TypeError, ValueError):
                    pass
                else:
                    if converted is not value and type(converted) is not type(value):
                        return normalize_json(converted)

        # NumPy's extended float scalar has no lossless Python scalar
        # equivalent.  Keep finite values as decimal strings and reserve null
        # for genuinely non-finite values.
        dtype = getattr(value, "dtype", None)
        if getattr(dtype, "kind", None) == "f" and getattr(dtype, "itemsize", 0) > 8:
            # Avoid math.isfinite: it narrows through float and mistakes a
            # finite value outside Python float's range for infinity.
            if math.isnan(value) or value == math.inf or value == -math.inf:
                return None
            return str(value)

        # NumPy bool_, ordinary numeric scalars, and zero-dimensional arrays do
        # not all register as the standard numeric ABCs.  Their item
        # conversion keeps this module usable without importing NumPy.
        item = getattr(value, "item", None)
        if callable(item):
            try:
                converted = item()
            except (TypeError, ValueError):
                pass
            else:
                if converted is not value and type(converted) is not type(value):
                    return normalize_json(converted)

        # A zero-dimensional array may fall back to tolist when item is not
        # available; non-zero-dimensional arrays were handled above.
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                converted = tolist()
            except (TypeError, ValueError):
                pass
            else:
                if converted is not value and type(converted) is not type(value):
                    return normalize_json(converted)

    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None

    return value


def dumps(value: Any, **kwargs: Any) -> str:
    """Serialize a record after normalization using standards-compliant JSON."""
    kwargs.pop("allow_nan", None)
    return json.dumps(normalize_json(value), allow_nan=False, **kwargs)
