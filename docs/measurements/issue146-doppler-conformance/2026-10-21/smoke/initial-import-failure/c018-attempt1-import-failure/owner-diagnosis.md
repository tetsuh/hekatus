# Owner-provided board-free diagnosis

The owner subsequently reproduced the failure using the pinned image with no device mapping, `--network none`, a read-only repository mount, and `python3 tools/issue146_device.py --help`.

```text
File "/work/tools/issue146_analysis.py", line 11, in <module>
    from scipy.stats import t as student_t
ModuleNotFoundError: No module named 'scipy'
```

The import chain was `tools.issue146_device` -> `tools.issue146` -> `tools.issue146_analysis` -> `scipy`. The pinned image lacks SciPy, while the host uv environment includes it. The same device-free `tools/issue137_device.py --help` command succeeded.

The owner established that the failed container exited during import and never opened a device. This is owner-provided later evidence, not a claim derived from the original run artifacts alone. The failed artifacts are preserved unchanged as import-failure diagnostics. The owner authorized board-free repairs followed by a fresh first smoke instance after the requested validation and publication gates; no reset or 180-second run is authorized.
