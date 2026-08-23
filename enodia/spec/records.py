"""Stage-to-stage records — the in-process form of the data-plane contract.

docs/dataplane.md warns that a reference implementation which passes bare
arrays across stage boundaries will have to reinvent those boundaries when
the stages become processes. So even in-process, data crosses a stage
boundary carrying the metadata the contract requires.

This is the minimal form. Records name their generation, and a consumer
checks that a batch it is given is internally consistent. What is *not* here,
and arrives with #15 when processes actually split: the shared-memory ring
buffer, the stream and tap policies, and matching a record's generation
against the table set a consumer holds — including the rule that
new-generation data is discarded until those tables are ready.

Two record kinds exist: raw RF for one transmit event, and its front-end
output — the int16 complex IQ of docs/dataplane.md T1, carried as two int16
planes because NumPy has no int16 complex (#6). The IQ record also names
the decimation ratio and the RF-time position of its first sample, so a
consumer cannot silently read D=4 data as D=8 or misplace it by half a
sample: both are properties of the record, checked by the consumer against
what it was configured for, in the same spirit as the generation tag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventHeader:
    """Metadata for one transmit event.

    The config ID and the generation counter are the single source of truth
    for synchronization (design.md §19): a consumer processes a record with
    the table set the record itself names, and discards it when those tables
    are not ready.
    """

    seq: int  # monotonically increasing sequence number
    config_id: str  # transmit-configuration identifier
    param_generation: int  # parameter-generation counter
    tx_event_index: int  # transmit-event number within the frame
    tx_type: str  # transmit-type tag; an open set, never an enum
    timestamp_ns: int  # acquisition timestamp


@dataclass(frozen=True)
class RFEventRecord:
    """Raw RF for one transmit event.

    data: int16, shape (n_channels, n_samples), sampled at the profile rate.

    Handing an array to a record transfers ownership of it: the array is
    marked read-only, because `frozen=True` stops the field being rebound
    but does nothing about the buffer behind it, and a consumer that mutates
    what it was given corrupts every other reader of the same frame.

    The array is **not** copied. Copying every record would double the peak
    memory of a frame, and the design has buffers handed in as partitions
    from outside rather than duplicated at each boundary (design.md §20).
    Read-only views are the contract; copies are not.
    """

    header: EventHeader
    data: np.ndarray

    def __post_init__(self) -> None:
        if self.data.dtype != np.int16:
            raise ValueError(f"RF payload must be int16, got {self.data.dtype}")
        if self.data.ndim != 2:
            raise ValueError(f"RF payload must be (channel, sample), got shape {self.data.shape}")
        if self.data.shape[1] < 2:
            # Fractional delays interpolate between neighbouring samples, so a
            # record carrying fewer than two is not delayable by any consumer.
            raise ValueError(f"RF payload needs at least 2 samples, got {self.data.shape[1]}")
        self.data.flags.writeable = False


@dataclass(frozen=True)
class IQEventRecord:
    """Front-end output for one transmit event: int16 complex IQ (dataplane T1).

    data: int16, shape (n_channels, n_samples, 2) — the last axis is (I, Q),
    the in-process form of the L1-resident int16 complex format (design.md
    §14). `complex()` promotes it to a complex floating array for the
    FP32-or-wider intermediate the delay stage runs at.

    decimation: the ratio D the front end applied; IQ samples sit fs/D apart.
    rf_offset: the RF-sample position of IQ sample 0 — IQ sample m stands for
    RF sample position m·D + rf_offset, so a consumer reads the IQ record at
    (p − rf_offset) / D for an RF position p. With an even number of FIR taps
    the group delay is a half sample and rf_offset is 0.5 (enodia.spec.frontend).

    The array is marked read-only and not copied, for the reasons
    RFEventRecord gives.
    """

    header: EventHeader
    data: np.ndarray
    decimation: int
    rf_offset: float

    def __post_init__(self) -> None:
        if self.data.dtype != np.int16:
            raise ValueError(f"IQ payload must be int16 (two planes), got {self.data.dtype}")
        if self.data.ndim != 3 or self.data.shape[-1] != 2:
            raise ValueError(
                f"IQ payload must be (channel, sample, 2) for (I, Q), got shape {self.data.shape}"
            )
        if self.data.shape[1] < 2:
            raise ValueError(f"IQ payload needs at least 2 samples, got {self.data.shape[1]}")
        if int(self.decimation) != self.decimation or self.decimation < 1:
            raise ValueError(f"decimation must be a positive integer, got {self.decimation}")
        if not math.isfinite(self.rf_offset):
            # A NaN offset makes every read position NaN, which the delay
            # stage turns into zero vectors — a silent black image, not an
            # error. Refuse it here, where it is still a record property.
            raise ValueError(f"rf_offset must be finite, got {self.rf_offset}")
        self.data.flags.writeable = False

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    def complex(self, dtype=np.complex64) -> np.ndarray:
        """The IQ as a complex array (n_channels, n_samples) in ``dtype``."""
        out = np.empty(self.data.shape[:2], dtype=dtype)
        out.real = self.data[..., 0]
        out.imag = self.data[..., 1]
        return out
