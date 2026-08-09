"""Stage-to-stage records — the in-process form of the data-plane contract.

docs/dataplane.md warns that a reference implementation which passes bare
arrays across stage boundaries will have to reinvent those boundaries when
the stages become processes. So even in-process, data crosses a stage
boundary carrying the metadata the contract requires.

This is the minimal form: the shared-memory ring buffer, the stream and tap
policies, and generation matching arrive when processes actually split (#15).
What is fixed here is that every record names its generation.
"""

from __future__ import annotations

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
        self.data.flags.writeable = False
