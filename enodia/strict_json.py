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
_NO_NUMPY_CONVERSION = object()


def _try_numpy_conversions(value: Any, *method_names: str) -> Any:
    """Return the first changed result from the named NumPy conversions."""
    for method_name in method_names:
        converter = getattr(value, method_name, None)
        if not callable(converter):
            continue
        try:
            converted = converter()
        except (TypeError, ValueError):
            continue
        if converted is not value and type(converted) is not type(value):
            return converted
    return _NO_NUMPY_CONVERSION


def _normalize_numpy(value: Any) -> Any:
    """Normalize NumPy values without importing NumPy."""
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim > 0:
        converted = _try_numpy_conversions(value, "tolist")
        if converted is not _NO_NUMPY_CONVERSION:
            return normalize_json(converted)

    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) == "f" and getattr(dtype, "itemsize", 0) > 8:
        # Avoid math.isfinite: it narrows through float and mistakes a finite
        # value outside Python float's range for infinity.
        if math.isnan(value) or value == math.inf or value == -math.inf:
            return None
        return str(value)

    converted = _try_numpy_conversions(value, "item", "tolist")
    if converted is not _NO_NUMPY_CONVERSION:
        return normalize_json(converted)
    return value


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
    if type(value).__module__.split(".", 1)[0] == "numpy":
        return _normalize_numpy(value)
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
