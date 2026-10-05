"""Run the Issue #100 default comparison in one device session."""

from __future__ import annotations

from pathlib import Path

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.run_matmul import run_custom_newton_schulz
from enodia.tt.bench.shapes import default_catalogue

ISSUE100_SHAPES = ("newton_schulz_L32_b8192", "newton_schulz_L16_b8192")
ISSUE100_DEVICE_ID = 0
ISSUE100_LAUNCHES = 1000
ISSUE100_COMPARISON_CONFIGS = (
    {
        "name": "new_default",
        "variant": "bf16",
        "matrix_block": 8,
        "double_buffer": True,
        "fuse_s": True,
        "math_fidelity": "HiFi3",
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    },
    {
        "name": "previous_default",
        "variant": "bf16-fp32state",
        "matrix_block": 4,
        "double_buffer": True,
        "fuse_s": True,
        "math_fidelity": "HiFi3",
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    },
)


def run_issue100_comparison(
    ttnn, device, *, repeats: int, stop_on_failure: bool = False
) -> list[dict]:
    """Run both Issue #100 configurations for both requested shapes.

    ``stop_on_failure`` is used by the combined acceptance driver so a failed
    row is terminal and no later device row is attempted.  The historical
    performance-only driver keeps its original continue-through-results
    behavior by default.
    """
    shapes = {
        shape.name: shape
        for shape in default_catalogue()
        if shape.name in ISSUE100_SHAPES
    }
    results: list[dict] = []
    for shape_name in ISSUE100_SHAPES:
        shape = shapes[shape_name]
        for config in ISSUE100_COMPARISON_CONFIGS:
            row = run_custom_newton_schulz(
                ttnn,
                device,
                shape,
                dtype_name="bfloat16",
                memory_name="l1",
                variant=config["variant"],
                iters=1,
                repeats=repeats,
                math_fidelity=config["math_fidelity"],
                fuse_s=config["fuse_s"],
                batch_reads=False,
                reload_r=False,
                matrix_block=config["matrix_block"],
                double_buffer=config["double_buffer"],
                input_memory="l1",
                r_memory="l1",
                x0_memory="l1",
                output_memory="dram",
                fp32_dest_acc_en=config["fp32_dest_acc_en"],
                dst_full_sync_en=config["dst_full_sync_en"],
                row_name=f"{shape_name}_{config['name']}",
            )
            row["shape_name"] = shape_name
            row["comparison_config"] = dict(config)
            if row["status"] == "ok":
                flops = row["flops_per_iteration"]
                row["tflops_p50_derived"] = flops / row["seconds_per_launch_p50"] / 1e12
                row["tflops_fastest_launch_derived"] = (
                    flops / min(row["seconds_per_launch_samples"]) / 1e12
                )
                row["tflops_provenance"] = {
                    "p50_derived": (
                        "flops_per_iteration / seconds_per_launch_p50 / 1e12"
                    ),
                    "fastest_launch_derived": (
                        "flops_per_iteration / min(seconds_per_launch_samples) / 1e12"
                    ),
                }
            results.append(row)
            print(
                shape_name,
                config["name"],
                row.get("status"),
                row.get("seconds_per_launch_p50"),
                row.get("seconds_per_launch_p99"),
                row.get("seconds_per_launch_p99_9"),
                row.get("tflops_p50_derived"),
                row.get("tflops_fastest_launch_derived"),
                flush=True,
            )
            if stop_on_failure and row.get("status") != "ok":
                return results
    return results


def main() -> int:
    import ttnn

    device = ttnn.open_device(device_id=ISSUE100_DEVICE_ID)
    try:
        results = run_issue100_comparison(ttnn, device, repeats=ISSUE100_LAUNCHES)
    finally:
        ttnn.close_device(device)

    output_path = Path("/out/issue100-same-run.json")
    payload = {
        "record_schema": "adr-0005-issue100-same-run-v1",
        "configuration_mode": "issue100-new-vs-previous-default-same-run",
        "device_id": ISSUE100_DEVICE_ID,
        "launches_per_row": ISSUE100_LAUNCHES,
        "watcher": False,
        "runner": "tools/newton_schulz_issue100_same_run.py",
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(strict_json_dumps(payload, indent=2) + "\n")
    return 0 if all(row.get("status") == "ok" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
