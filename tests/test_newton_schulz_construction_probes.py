import json
from dataclasses import fields

import pytest

from tools import newton_schulz_bringup as bringup

PROBE_NAMES = ("P0", "P1", "P2", "P3a", "P3b", "P4", "P5", "P6", "P7", "P8")


class _FakeTensor:
    _next_address = 100

    def __init__(self, values, dtype):
        self.values = values
        self.dtype = dtype
        self.address = self._next_address
        type(self)._next_address += 1
        self.deallocated = False

    def buffer_address(self):
        return self.address


class _FakeDevice:
    class _Grid:
        x = 1
        y = 1

    def compute_with_storage_grid_size(self):
        return self._Grid()


class _FakeRuntimeArgs:
    def __init__(self):
        self.values = {}

    def __getitem__(self, x):
        return _FakeRuntimeRow(self, x)


class _FakeRuntimeRow:
    def __init__(self, args, x):
        self.args = args
        self.x = x

    def __setitem__(self, y, values):
        self.args.values[(self.x, y)] = values


class _FakeCoreCoord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeCoreRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeCoreRangeSet:
    def __init__(self, ranges):
        self.ranges = ranges


class _FakeKernelDescriptor:
    class SourceType:
        FILE_PATH = "file-path"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)


class _FakeFormatDescriptor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeCBDescriptor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeProgramDescriptor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeAccessorArgs:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_compile_time_args(self):
        return [self.tensor.address + 1000]


class _FakeTtnn:
    bfloat16 = "bfloat16"
    float32 = "float32"
    TILE_LAYOUT = "tile"
    L1_MEMORY_CONFIG = "l1"
    KernelDescriptor = _FakeKernelDescriptor
    RuntimeArgs = _FakeRuntimeArgs
    CoreCoord = _FakeCoreCoord
    CoreRange = _FakeCoreRange
    CoreRangeSet = _FakeCoreRangeSet
    TensorAccessorArgs = _FakeAccessorArgs
    ReaderConfigDescriptor = _FakeConfig
    WriterConfigDescriptor = _FakeConfig
    ComputeConfigDescriptor = _FakeConfig
    CBFormatDescriptor = _FakeFormatDescriptor
    CBDescriptor = _FakeCBDescriptor
    TileDescriptor = _FakeConfig
    ProgramDescriptor = _FakeProgramDescriptor
    Shape = tuple

    def __init__(self):
        self.opened_device_ids = []
        self.closed_devices = []
        self.generic_calls = []
        self.deallocated = []

    def Tensor(self, values, dtype):
        return _FakeTensor(values, dtype)

    def to_layout(self, tensor, layout):
        tensor.layout = layout
        return tensor

    def to_device(self, tensor, device, memory_config):
        tensor.device = device
        tensor.memory_config = memory_config
        return tensor

    def allocate_tensor_on_device(self, shape, dtype, layout, device, memory_config):
        tensor = _FakeTensor(shape, dtype)
        tensor.device = device
        tensor.layout = layout
        tensor.memory_config = memory_config
        return tensor

    def open_device(self, device_id):
        self.opened_device_ids.append(device_id)
        return _FakeDevice()

    def close_device(self, device):
        self.closed_devices.append(device)

    def generic_op(self, tensors, program):
        self.generic_calls.append((tensors, program))

    def synchronize_device(self, device):
        return None

    def deallocate(self, tensor):
        tensor.deallocated = True
        self.deallocated.append(tensor)


def _probe_fields(probe):
    ignored = {"name", "purpose"}
    return {
        field.name: getattr(probe, field.name)
        for field in fields(probe)
        if field.name not in ignored
    }


def _changed_fields(left, right):
    return {
        name for name, value in _probe_fields(left).items() if value != _probe_fields(right)[name]
    }


def _dispatch_record():
    return bringup._construction_dispatch_record(
        bringup.construction_probe_for("P8"), device_id=0, elapsed=0.25
    )


def test_probe_sequence_is_host_only_and_has_the_fixed_dispatch_shape():
    assert tuple(bringup.CONSTRUCTION_PROBES) == PROBE_NAMES
    for name in PROBE_NAMES:
        probe = bringup.construction_probe_for(name)
        assert probe.batch == 1
        assert probe.core_count == 1
        assert probe.core_coordinates == ((0, 0),)
        assert probe.core_ranges == (((0, 0), (0, 0)),)
        assert probe.tiles_per_core == 1
        assert probe.tile_count == 0
        assert probe.iterations == 8
        assert probe.fp32_dest_acc_en is True
        assert probe.output_dtypes == ("float32", "float32")
        assert probe.reader_accessor_count == 6
        assert probe.writer_accessor_count == 2
        assert probe.reader_runtime_arg_count == 8
        assert probe.writer_runtime_arg_count == 4
        assert probe.compute_compile_args == (1,)
        assert probe.compute_runtime_args == ()
        dispatch = bringup.construction_dispatch_configuration(probe)
        assert dispatch.reader_tile_count == 0
        assert dispatch.writer_tile_count == 0
        assert dispatch.compute_compile_args == (0,)
        assert probe.kernel_order == ("reader", "writer", "compute")
        assert probe.semaphores == ()
        assert probe.external_timeout_seconds == 60
        assert len(probe.cb_formats) == 25
        assert len(probe.cb_page_sizes) == 25
        assert probe.cb_page_sizes == tuple(
            bringup.TILE_BYTES_FLOAT32 if fmt == "float32" else bringup.TILE_BYTES_BFLOAT16
            for fmt in probe.cb_formats
        )
        assert len(probe.active_cb_indices) in (18, 24, 25)
        assert all(path.is_file() for path in bringup.construction_source_paths(probe))


def test_p0_to_p2_isolates_input_and_descriptor_construction_changes():
    p0, p1, p2 = (bringup.construction_probe_for(name) for name in ("P0", "P1", "P2"))
    assert _changed_fields(p0, p1) == {"input_dtypes"}
    assert [
        index
        for index, (before, after) in enumerate(zip(p0.input_dtypes, p1.input_dtypes))
        if before != after
    ] == [1, 3]
    assert p0.input_dtypes[0] == p1.input_dtypes[0] == "bfloat16"
    assert [p0.input_dtypes[index] for index in (2, 4, 5)] == [
        p1.input_dtypes[index] for index in (2, 4, 5)
    ]
    assert _changed_fields(p1, p2) == {"cb_formats", "cb_page_sizes"}
    assert [
        index
        for index, (before, after) in enumerate(zip(p1.cb_formats, p2.cb_formats))
        if before != after
    ] == [14, 15, 20, 21]
    assert [
        index
        for index, (before, after) in enumerate(zip(p1.cb_page_sizes, p2.cb_page_sizes))
        if before != after
    ] == [14, 15, 20, 21]
    assert p2.cb_formats[14:16] == ("float32", "float32")
    assert p2.cb_formats[20:22] == ("bfloat16", "bfloat16")
    assert p2.cb_page_sizes[14:16] == (bringup.TILE_BYTES_FLOAT32,) * 2
    assert p2.cb_page_sizes[20:22] == (bringup.TILE_BYTES_BFLOAT16,) * 2


def test_p3_pair_keeps_the_same_noop_source_and_changes_only_active_cb_indices():
    p3a, p3b = (bringup.construction_probe_for(name) for name in ("P3a", "P3b"))
    assert _changed_fields(p3a, p3b) == {"active_cb_indices"}
    assert p3a.active_cb_indices == bringup.STAGE_61_ACTIVE_CB_INDICES
    assert p3b.active_cb_indices == bringup.STAGE_62_ACTIVE_CB_INDICES
    assert len(p3a.active_cb_indices) == 18
    assert len(p3b.active_cb_indices) == 24
    assert bringup.construction_source_paths(p3a) == bringup.construction_source_paths(p3b)
    for source in bringup.construction_source_paths(p3a):
        text = source.read_text()
        assert "matmul" not in text
        assert "copy_tile" not in text
        assert "binary_op" not in text


def test_p4_to_p8_changes_only_the_next_approved_boundary():
    p3b, p4, p5, p6, p7, p8 = (
        bringup.construction_probe_for(name) for name in ("P3b", "P4", "P5", "P6", "P7", "P8")
    )
    assert _changed_fields(p3b, p4) == {"reader_source"}
    assert _changed_fields(p4, p5) == {"compute_source"}
    assert _changed_fields(p5, p6) == {"cb_formats", "cb_page_sizes"}
    assert _changed_fields(p6, p7) == {"reader_source", "active_cb_indices"}
    assert _changed_fields(p7, p8) == {"compute_source"}
    assert [
        index
        for index, (before, after) in enumerate(zip(p5.cb_formats, p6.cb_formats))
        if before != after
    ] == [13]
    assert [
        index
        for index, (before, after) in enumerate(zip(p5.cb_page_sizes, p6.cb_page_sizes))
        if before != after
    ] == [13]
    assert p5.cb_formats[13] == "bfloat16"
    assert p6.cb_formats[13] == "float32"
    assert p5.cb_page_sizes[13] == bringup.TILE_BYTES_BFLOAT16
    assert p6.cb_page_sizes[13] == bringup.TILE_BYTES_FLOAT32
    assert p6.active_cb_indices == bringup.STAGE_62_ACTIVE_CB_INDICES
    assert p7.active_cb_indices == p8.active_cb_indices == bringup.STAGE_67_ACTIVE_CB_INDICES


def test_probe_record_is_serializable_and_makes_no_numerical_or_board_claim():
    record = bringup.construction_probe_record("P8")
    assert record["status"] == "host-configured"
    assert record["board_run"] is False
    assert record["numerical_acceptance"] is False
    assert record["tile_count"] == 0
    assert record["compute_compile_args"] == [1]
    assert record["compute_runtime_args"] == []
    assert record["dispatch_zero_work"] == {
        "reader_tile_count": 0,
        "writer_tile_count": 0,
        "compute_compile_args": [0],
    }
    assert record["semaphores"] == []
    assert record["core_range_count"] == 1
    assert record["core_ranges"] == [[[0, 0], [0, 0]]]
    assert record["cb_descriptor_count"] == 25
    assert record["cb_page_count"] == 4
    assert record["cb_descriptor_total_bytes"] == 352256
    json.loads(json.dumps(record))


def test_dispatch_record_json_round_trip_is_accepted_without_opening_a_device():
    record = json.loads(json.dumps(_dispatch_record()))

    assert bringup.is_successful_construction_dispatch_record(record)


def test_host_configured_and_failed_records_are_rejected_by_dispatch_predicate():
    host_record = bringup.construction_probe_record("P8")
    failed_record = dict(host_record, status="fail", error="dispatch failed")

    assert not bringup.is_successful_construction_dispatch_record(host_record)
    assert not bringup.is_successful_construction_dispatch_record(failed_record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "dispatch"),
        ("board_run", 1),
        ("device_id", True),
        ("device_id", -1),
        ("device_id", 1.0),
        ("dispatch_elapsed_s", True),
        ("dispatch_elapsed_s", float("nan")),
        ("dispatch_elapsed_s", float("inf")),
        ("dispatch_elapsed_s", -1.0),
    ],
)
def test_dispatch_predicate_rejects_malformed_or_typed_dispatch_fields(field, value):
    record = _dispatch_record()
    record[field] = value

    assert not bringup.is_successful_construction_dispatch_record(record)


@pytest.mark.parametrize(
    "zero_work",
    [
        None,
        {},
        {"reader_tile_count": "0", "writer_tile_count": 0, "compute_compile_args": [0]},
        {"reader_tile_count": 0, "writer_tile_count": True, "compute_compile_args": [0]},
        {"reader_tile_count": 0, "writer_tile_count": 0, "compute_compile_args": [1]},
        {"reader_tile_count": 0, "writer_tile_count": 0, "compute_compile_args": [True]},
        {"reader_tile_count": 0, "writer_tile_count": 0, "compute_compile_args": (0,)},
        {
            "reader_tile_count": 0,
            "writer_tile_count": 0,
            "compute_compile_args": [0],
            "unexpected": 0,
        },
    ],
)
def test_dispatch_predicate_rejects_malformed_or_altered_zero_work(zero_work):
    record = _dispatch_record()
    record["dispatch_zero_work"] = zero_work

    assert not bringup.is_successful_construction_dispatch_record(record)


def test_dispatch_builder_uses_zero_work_args_and_all_indexed_cb_descriptors():
    ttnn = _FakeTtnn()
    probe = bringup.construction_probe_for("P8")
    program, inputs, outputs = bringup.build_construction_probe_program(ttnn, _FakeDevice(), probe)
    try:
        assert [tensor.dtype for tensor in inputs] == list(probe.input_dtypes)
        assert [tensor.dtype for tensor in outputs] == ["float32", "float32"]
        reader, writer, compute = program.kernels
        assert reader.runtime_args.values[(0, 0)][-2:] == [0, 0]
        assert writer.runtime_args.values[(0, 0)][-2:] == [0, 0]
        assert compute.compile_time_args == [0]
        assert compute.runtime_args == []
        assert len(program.cbs) == 25
        for index, descriptor in enumerate(program.cbs):
            format_descriptor = descriptor.format_descriptors[0]
            assert format_descriptor.buffer_index == index
            assert format_descriptor.data_format == getattr(ttnn, probe.cb_formats[index])
            assert format_descriptor.page_size == probe.cb_page_sizes[index]
    finally:
        bringup._deallocate_tensors(ttnn, [*inputs, *outputs])


def test_authorized_dispatch_opens_selected_device_and_releases_all_tensors():
    ttnn = _FakeTtnn()
    result = bringup.dispatch_construction_probe(
        "P0",
        allow_device_dispatch=True,
        device_id=3,
        ttnn_module=ttnn,
    )
    assert result["status"] == "dispatched"
    assert result["board_run"] is True
    assert result["device_id"] == 3
    assert ttnn.opened_device_ids == [3]
    assert len(ttnn.generic_calls) == 1
    tensors, program = ttnn.generic_calls[0]
    assert len(tensors) == 8
    assert program.kernels[0].runtime_args.values[(0, 0)][-1] == 0
    assert program.kernels[1].runtime_args.values[(0, 0)][-1] == 0
    assert program.kernels[2].compile_time_args == [0]
    assert all(tensor.deallocated for tensor in tensors)
    assert len(ttnn.closed_devices) == 1


def test_dispatch_is_explicitly_gated_and_keeps_the_host_injection_seam():
    ttnn = _FakeTtnn()
    with pytest.raises(PermissionError, match="disabled"):
        bringup.dispatch_construction_probe("P0", ttnn_module=ttnn)
    assert ttnn.opened_device_ids == []
    seen = []
    result = bringup.dispatch_construction_probe(
        "P0",
        allow_device_dispatch=True,
        dispatcher=lambda probe: seen.append(probe.name) or "injected-result",
    )
    assert result == "injected-result"
    assert seen == ["P0"]
