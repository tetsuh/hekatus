"""Derive a new instrument from hash-pinned resident sources, never edit them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).resolve().parent
BASE = SOURCE.parent / "kernels"
TABLES_HEADER = "resident_tables.h"
PROTOCOL_HEADER = "protocol.hpp"
DEVICE_HEADER = "device.hpp"
SOURCE_MANIFEST = "sources.json"
OWNED_OUTPUTS = (ROOT / "out/issue120-production-build/kernels",
                 Path("/out/issue120-production-device-check/kernels"))
BASE_HASHES = {
    "resident_producer.cpp": "ebf3eac498a7f9d5ef835f3c9fd81410b46a7bb4c57cda2e4905c10ff25ef976",
    "resident_consumer.cpp": "2ef8bef2e0600069336bd0b67db56e0581112254a283510b2cde0a368c21529a",
    TABLES_HEADER: "090a48f3296c404bfce5d6662ce880ac3048aa0bf40d28c6fbd6258efd1d925c",
}


def _replace(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("versioned resident hook is absent or ambiguous")
    return text.replace(old, new, 1)


def derive(role: str) -> str:
    path = BASE / f"resident_{role}.cpp"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_HASHES[path.name]:
        raise ValueError("resident base changed; a new adapter version is required")
    text = raw.decode()
    text = _replace(text, '#include "resident_tables.h"',
                    '#include "resident_tables.h"\n#include "device.hpp"')
    start = text.index("struct WrapTrackedClock {")
    end = text.index("};", start) + 2
    text = text[:start] + "using WrapTrackedClock = issue120_production::Clock;" + text[end:]
    text = _replace(text, "    WrapTrackedClock clock;\n    clock.initialize();\n", "")
    text = _replace(text, "void kernel_main() {", "void kernel_main() {\n"
                    "    WrapTrackedClock clock;\n    clock.initialize();")
    for original, replacement in (("noc_async_read_barrier();", "reads_complete(clock);"),
                                  ("noc_async_write_barrier();", "writes_complete(clock);"),
                                  ("noc_async_atomic_barrier();", "atomics_complete(clock);")):
        text = text.replace(original, "issue120_production::" + replacement)
    text = _replace(text, "    const auto ring = TensorAccessor(ring_args, ring_address);\n", "")
    if role == "producer":
        text = _replace(text, "const std::uint64_t consumer_noc = control.get_noc_addr(0);",
                        "const std::uint64_t consumer_noc = get_noc_addr(\n"
                        "        get_arg_val<uint32_t>(issue120_production::producer_arg::peer_x),\n"
                        "        get_arg_val<uint32_t>(issue120_production::producer_arg::peer_y), 0);")
        text = _replace(text, "    const std::uint64_t run_start = clock.read();",
                        "    issue120_production::wait_start(clock, run_budget_ticks);\n"
                        "    const std::uint64_t run_start = clock.read();")
        text = _replace(text,
                        "noc_async_write_page(ready_count % ring_pages, ring, get_read_ptr(cb_scratch));",
                        "issue120_production::write_ring(ready_count % ring_pages, ring_address,\n"
                        "            get_read_ptr(cb_scratch));")
        text = _replace(text, "    summary[11] = failure_code != 0;",
                        "    summary[11] = failure_code != 0;\n    summary[12] = clock.ambiguous;")
    elif role == "consumer":
        text = _consumer(text)
    else:
        raise ValueError("unsupported resident role")
    return "// Versioned Issue120 adapter; hash-pinned Issue118 work follows.\n" + text


def _consumer(text: str) -> str:
    text = _replace(text, "    const auto producer_anchor = TensorAccessor(producer_anchor_args, producer_anchor_address);\n", "")
    text = _replace(text, "const std::uint64_t producer_anchor_noc = producer_anchor.get_noc_addr(0);",
                    "const std::uint64_t producer_anchor_noc = get_noc_addr(\n"
                    "        get_arg_val<uint32_t>(issue120_production::consumer_arg::peer_x),\n"
                    "        get_arg_val<uint32_t>(issue120_production::consumer_arg::peer_y), 0);")
    text = _replace(text, "    const auto tables = TensorAccessor(tables_args, get_arg_val<std::uint32_t>(12));",
                    "    const auto tables = TensorAccessor(tables_args, get_arg_val<std::uint32_t>(12));\n"
                    "    constexpr auto progress_args = TensorAccessorArgs<tables_args.next_compile_time_args_offset()>();\n"
                    "    const auto progress = TensorAccessor(progress_args,\n"
                    "        get_arg_val<uint32_t>(issue120_production::consumer_arg::progress));")
    text = _replace(text, "    const std::uint64_t run_start = clock.read();",
                    "    const std::uint64_t run_start = clock.read();\n"
                    "    issue120_production::Observer<decltype(progress)> observation(progress, frame_count, clock);")
    text = _replace(text, "        const std::uint64_t start = clock.read();",
                    "        const auto observed = observation.before_frame(clock);\n"
                    "        const std::uint64_t start = clock.read();")
    text = _replace(text, "        timestamp_page[3] = static_cast<std::uint32_t>(elapsed);",
                    "        timestamp_page[3] = static_cast<std::uint32_t>(elapsed);\n"
                    "        observation.record(timestamp_page, frames_consumed, observed, start, clock);")
    text = _replace(text, "        noc_semaphore_inc(free_noc, 1);\n"
                    "        issue120_production::atomics_complete(clock);\n"
                    "        if (error_flag != 0)",
                    "        noc_semaphore_inc(free_noc, 1);\n"
                    "        issue120_production::atomics_complete(clock);\n"
                    "        observation.after_frame(frames_consumed, clock);\n"
                    "        if (error_flag != 0)")
    text = _replace(text, "    summary[0] = consumer_empty_count;",
                    "    observation.finish(clock);\n    observation.summary(summary, frames_consumed, clock);\n"
                    "    summary[0] = consumer_empty_count;")
    # Publish completion only after the original final summary transfer drained.
    end = text.rfind("}")
    return text[:end] + "    observation.completed(clock);\n" + text[end:]


def _owned_output(output: Path) -> Path:
    for owned in OWNED_OUTPUTS:
        if output.absolute() == owned and output.resolve() == owned:
            return owned
    raise ValueError("kernel generation requires a fixed program-owned destination")


def generate(output: Path) -> dict[str, Path]:
    output = _owned_output(output)
    names = ("production_producer.cpp", "production_consumer.cpp", TABLES_HEADER,
             PROTOCOL_HEADER, DEVICE_HEADER, SOURCE_MANIFEST)
    for name in names:
        target = output / name
        if target.is_symlink() or target.resolve() != target.absolute():
            raise ValueError("generated kernel target is redirected")
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for role in ("producer", "consumer"):
        target = output / f"production_{role}.cpp"
        target.write_text(derive(role))
        outputs[role] = target
    for name, source in ((TABLES_HEADER, BASE / TABLES_HEADER),
                         (PROTOCOL_HEADER, SOURCE / PROTOCOL_HEADER), (DEVICE_HEADER, SOURCE / DEVICE_HEADER)):
        raw = source.read_bytes()
        if name in BASE_HASHES and hashlib.sha256(raw).hexdigest() != BASE_HASHES[name]:
            raise ValueError("resident support header changed")
        (output / name).write_bytes(raw)
    manifest = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in sorted(names) if name != SOURCE_MANIFEST}
    (output / SOURCE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    generate(OWNED_OUTPUTS[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
