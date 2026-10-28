# Resident table preload and swap — Issue #119 mechanism

This is an extension of the #12/#118 instrument under the existing
[discard contract](dataplane.md#generation-mismatch-discard-never-substitute),
not an Ethernet receiver or derivative-generation implementation. Tables are
synthetic inputs to the fixed work, not beamformer derivatives. This PR is
board-free; measurement acceptance criteria remain for a subsequent PR.
No new architecture or contract decision is introduced: the mechanism follows
Issue #119 and design §12(3)/§19, within ADR-0005's instrument scope.

## Configuration and wire format

`ResidentConfig` has optional `profile_tables` tuples
`(config_id, generation, ready_at_attempted_frame)`, `switch_schedule` tuples
`(attempted_frame, config_id, generation)`, `initial_identity`, `table_bytes`,
and `swap_strategy` (`copy_on_demand` or `prefetch`). Profile count and switch
count are the lengths of those inputs. Spacing is explicit, not a kernel
constant. Every profile occupies the same positive multiple of 4096 bytes.
Identities are unique among profile rows; generation alone need not be unique.
A schedule may name an absent or delayed identity for a discard positive control.

The defaults retain the legacy path: no profile or schedule, no table allocation,
no checksum extension, and unchanged output fields. Explicit profile inputs
without switches enable the table mechanism for a same-program no-switch control.

With the mechanism enabled, each ring page begins with four uint32 words:
config ID, generation, attempted frame index, schedule segment (0 initially,
1..K for switches). The remaining words retain the synthetic payload. Fixed
work reconstructs the four displaced synthetic words from the attempted frame,
reads the selected L1 table, and resets its accumulator for each processed frame.
The Python checksum reproduces uint32 arithmetic, including overflow. No
identity-discarded frame writes a timestamp. Producer overflow can skip attempted
frame indices; the producer advances its schedule even across such drops.

One DRAM metadata page carries:

- four-word header: profile count, switch count, initial config, initial generation;
- four words per profile: config, generation, absolute ready frame, reserved zero;
- three words per switch: attempted frame, config, generation; zero padding.

Tables occupy a separate row-major interleaved DRAM tensor, with a 4096-byte
accessor page. Both buffers are uploaded and synchronized before launch. Each
kernel reads metadata once into its own L1 page. The producer stamps identities;
the consumer scans all profile descriptors and requires both identity fields and
readiness. The consumer frees discarded slots as well as processed slots, with
no catch-up and no substitution of its previous L1 copy. GenericOp receives
all table inputs before the legacy IO tensors, keeping `consumer_stats` last
as its output specification; without tables the legacy IO order is unchanged.

Copy-on-demand uses one L1 table slot. Prefetch uses two: the previous attempted
frame, when processed rather than discarded, may copy the next scheduled ready
profile into the inactive slot. A missed
lookahead (for example producer overflow) or delayed readiness takes the demand
path. A cached copy never overrides readiness. Every copy reads exactly the
configured page count and waits for each page's read barrier. Demand and
prefetch copies are both inside the per-frame timer; reaching the existing work
budget retains failure code 4 and producer cancellation. This does not claim a
measured copy duration or prove that a particular budget will fit.
With prefetch, copy cost appears in the preceding processed frame's interval,
not just the interval of the new segment's first processed frame.

## Accounting and bounds

The ledger includes both metadata pages, a consumer switch-stats page, and one
or two complete L1 table copies, in addition to the existing ring, CBs, and
semaphores. The existing conservative L1 ceiling applies to the total. Metadata
and switch summaries must each fit one 1024-word page; preloaded table storage
is bounded to 1 GiB. Rejections occur before TTNN import/device open and retain
the configuration and rejection reason.

Consumer summary words 0..17 retain their meanings. With tables enabled, word
18 is identity discards, 19 is retired pages, and words 20 onward contain four
words per segment: discards, interval low, interval high, validity (0 no processed
frame, 1 first completion without predecessor, 2 valid interval). Timestamp rows
retain words 0..3 and add attempted frame, config, generation, segment at 4..7.
The host checks stamped identities against the schedule, readiness, increasing
frame indices, every frame checksum, and card switch intervals against downloaded
completion timestamps. A checksum mismatch marks the record as an error.
Discarded frames do not update fixed-work min/max/valid or receive per-frame
work-budget checks; only the run budget applies. Thus an all-discard run has
`work_valid=0`, and a readiness delay of D discards removes D work samples.

The optional `table_swap` extension contains total/initial/per-switch discards,
retired count, per-switch intervals, checked-frame and mismatch counts, verification
status, and SHA-256 of `(frame, config, generation, checksum)` words. The normal
histogram remains beside it and includes intervals spanning discards; the
switch interval is specifically the previous completion to the first processed
frame in a schedule segment. If no such frame or predecessor exists, it is null,
not zero or a fabricated interval. Each switch additionally carries host-derived
`pre_switch_interval`: the completion interval ending at the last processed frame
before that scheduled switch, in device ticks, or null without two preceding
completions. It uses the last actual processed frame even if an intervening
segment contained only discards. The field is optional when reading older
records; no kernel timestamp row or summary word is added for this observation.
The parameters retain the complete schedule, table configuration, and L1 ledger.
Legacy records need no new fields.

## Power-trace capture order

For non-Watcher runs, the shared loaded-run capture block waits for
program-completion coverage,
then captures a fresh run-end candidate, waits for a sample at or after that
candidate, captures final `run_end`, and freezes the trace. Completion may
already be bracketed after result decoding; using it alone can leave the frozen
tail older than the nominal sampler interval because telemetry acquisition adds
overhead. The second wait reduces that tail to polling (0.1 s), write and host
scheduling latency without changing the one-interval coverage predicate or the
UTC/monotonic program span used for weighted AICLK. Each wait retains the
`interval * 1.5 + 1.0` deadline; exhaustion of the additional candidate wait still
freezes the available bytes and records incomplete coverage rather than rejecting
the run for that timeout. The initial completion-bracketing requirement remains
unchanged. Watcher, sampler-off and legacy unloaded capture paths are unchanged.
The supervisory cap is unchanged. PR #156 routes table-enabled controls through
this same block, so that capture order also applies to them after integration.

## Board-free demonstration

```sh
uv run pytest -q tests/test_resident_tables.py tests/test_resident_table_integration.py
uv run python enodia/tt/bench/run_resident.py --help
```

The CLI accepts repeatable `--profile-table CONFIG:GENERATION:READY_FRAME` and
`--switch FRAME:CONFIG:GENERATION`, plus `--initial-identity`, `--table-bytes`,
and `--swap-strategy`. The tests exercise delayed readiness and resumption,
partial-identity mismatches, corruption, descriptor construction, and preloading
using a fake TTNN API. They do not open a device.

## Pinned v0.75.0 feasibility

Source-confirmed APIs (tag `v0.75.0`):

- Host upload: `ttnn.Tensor(numpy_uint32, ttnn.uint32)` and `ttnn.to_device(...,
  memory_config=ttnn.DRAM_MEMORY_CONFIG)`; existing instrument/load code already
  uses these tensor APIs. See upstream
  [NumPy Tensor constructor](https://github.com/tenstorrent/tt-metal/blob/v0.75.0/ttnn/cpp/ttnn-nanobind/pytensor.cpp#L636-L730),
  [to_device binding](https://github.com/tenstorrent/tt-metal/blob/v0.75.0/ttnn/cpp/ttnn-nanobind/operations/core.cpp#L136-L160),
  and [generic-op example](https://github.com/tenstorrent/tt-metal/blob/v0.75.0/tests/ttnn/unit_tests/operations/debug/test_generic_op.py).
- Accessors: `ttnn.TensorAccessorArgs(...).get_compile_time_args()` on the host,
  paired with device `TensorAccessorArgs`, `next_compile_time_args_offset()`, and
  `TensorAccessor`; see the example above and
  [accessor arguments](https://github.com/tenstorrent/tt-metal/blob/v0.75.0/tt_metal/hw/inc/api/tensor/tensor_accessor_args.h).
- DRAM-to-L1 copies: `noc_async_read_page` and `noc_async_read_barrier` in
  [dataflow_api.h](https://github.com/tenstorrent/tt-metal/blob/v0.75.0/tt_metal/hw/inc/api/dataflow/dataflow_api.h).
  `CBDescriptor.total_size` supplies the fixed table slot allocation, and
  `get_write_ptr` supplies its local destination address.
- Schedule delivery: the metadata buffer address is a `RuntimeArgs` word of
  the producer's `KernelDescriptor`. The same metadata accessor is compiled
  into both resident kernels. The generic-op example constructs runtime args,
  kernel descriptors and circular buffers, then calls `ttnn.generic_op` once.
  Switch execution is entirely inside the resident loop; no host CQ completion
  wait or mid-run command is used. #120's second-CQ wait limitation is avoided.

Unverified without the board: kernel JIT (including header/toolchain integration),
actual NOC access/coherency, hardware checksum and counter agreement, copy latency,
work-budget fit, Watcher behavior, and timing/discard measurements. The pinned
image `--help` check proves Python import/CLI compatibility only. The first board
run requires owner authorization, Watcher, a small frame count and a 60-second
cap; it is diagnostic, not timing evidence. No device run belongs to this PR.
