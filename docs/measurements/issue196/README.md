# Issue #196: pinned attempt-runtime layout and device-free compiler gate

T references tt-metal v0.75.0 commit d9a68815f5fcf08a5bfbffb6f1f811823fba8edd.

## Invariants and supported method

No board/tt-smi/device/reset/firmware/cable/host-setting access is included. The
cards are off. Official test/kernel source, official binaries, image and library
bytes are unchanged. Historical A/Bv1/G0 evidence is never rewritten.

The pinned root selection is explicit: `T:tt_metal/llrt/rtoptions.cpp:315-345`
uses TT_METAL_RUNTIME_ROOT (or SetRootDir/CWD fallback); kernel source lookup is
separate (`T:tt_metal/impl/kernels/kernel.cpp:43-91`, rtoptions.cpp:430-437).
Normal JIT include roots use that runtime root, plus the kernel's own parent
and explicit config paths (`T:tt_metal/jit_build/build.cpp:102-110,300-318,570-577`;
kernel.cpp:340-357). The official fixture supplies no extra compiler include
root. Bv1 set runtime root to the wheel and only kernel lookup to the source
checkout, so source-root-relative tests/... includes failed after .cpp discovery.

`issue30_runtime.py` now creates **ATTEMPT/runtime** under the fresh capture:

| Entry | Source / permission |
|---|---|
|tt_metal, runtime, build, optional ttnn|links to immutable audited pinned image assets|
|tests|link to unchanged readonly pinned source/tests|
|generated/fabric|fresh owned writable output directory, never an image write|
|runtime-layout.json|attempt-root provenance, not a hardware observation|

Set TT_METAL_RUNTIME_ROOT to this program-owned view and retain the normal
TT_METAL_KERNEL_PATH source-file search. No source-root `-I` is individually
added to the official kernel config/compiler. The timeout encloses bootstrap and
exec of the unchanged official ELF. Non-root, readonly image/source/assets,
network-none, no-new-privileges, dropped capabilities and private scratch remain.
Only attempt output is newly writable; native JIT cache is ATTEMPT/cache.

This follows [#173](https://github.com/tetsuh/hekatus/issues/173)/#120's fixed,
audited pinned-image root rather than accepting arbitrary/inherited roots.
The existing #120 implementation validates image asset inventories; it does not
supply a tests subtree missing from that wheel. The new composition reuses its
validator and exact assets, not another runtime/library or an edited kernel.

## Export warnings: fixed output role, no waiver

Control-plane exports use `get_root_dir()/generated/fabric`, **not logs path**
(`T:tt_metal/fabric/control_plane.cpp:597-614`). Serializers create the parent
(fabric_host_utils.cpp:181-185,221-225). New runtime root directs that normal path
to fresh attempt output. Old Bv1's two warnings remain unchanged and nonbaseline;
no new exception/allowlist is introduced. A device-free writeability probe at
the identical generated role and the regular root selection verify this layout;
no device/TopologyMapper invocation is needed to prove filesystem permissions.
Those caught export failures concern metadata/visualization, not the direct RX
compile exception, but must still be absent for an accepted new smoke run.

## Compiler inventory and provenance

`issue196_compile_manifest.json` freezes the full failed RX argv/search roots
from Bv1 stdout,22 saved kernel_args.csv processor specializations and their
unchanged generated headers, the official smoke source and registered Watcher
kernel names. `offline-inputs.tar.gz` contains only canonical regular generated
support files; original Bv1 cache and records are untouched.

Seven source families,24 translation-unit cases:

| Family | Cases / origin |
|---|---|
|fabric_erisc_router.cpp|16 actual active/subordinate ERISC specializations|
|cq_dispatch.cpp|1 actual BRISC|
|cq_prefetch.cpp|1 actual BRISC|
|cq_dispatch_subordinate.cpp|1 actual NCRISC|
|cq_dispatch_subordinate_compute.cpp|3 actual TRISC processors|
|tt_fabric_1d_rx.cpp|1 full argv captured at failure|
|tt_fabric_1d_tx.cpp|1 source-derived sibling of RX; never compiled in Bv1|

Tests fix equality between source-fixture paths plus retained registered names
and target families, all24 specialization IDs and nonbenchmark options. HAL
processor includes/architecture/stack flags and optimization are reconstructed
from pinned source, not claimed to be independently captured full argv for every
successful Bv1 compilation. Router opt is O3 (VC1 inactive), dispatch-S Os,
compute O3, worker dispatch/prefetch/RX/TX O2. No extra source-root include patch.

**Limits:** TX uses the official shared RX/TX CTA vector and unchanged config;
its FULL_KERNEL_NAME is an explicit role surrogate, not a future JIT key. Future
boot/mesh/dispatch/architecture option changes require renewed inventory/check.
**OPEN:** base firmware is distinct from the seven registered kernel sources.
Saved B cache contains10 firmware object files; this checker does not recompile
those firmware translation units or prove a precompiled-image cache hit. The
pinned firmware asset bytes remain unchanged; normal build-key/precompiled lookup
is source-defined in build_env_manager.cpp:164-183,249-263. Object compile/
dependency success is not full firmware/kernel ELF linking, hardware dispatch,
credit/handshake, payload correctness or a guarantee for the whole cold-JIT path.

## RED/GREEN and qualification

- Behavioral RED before implementation: test_layout_root_not_just_kernel_search_path
  rejected the old request's wheel-only root.
- Device-free RED: old layout using the retained RX argv fails with the same
  kernel_utils.hpp missing include (compiler1,0.039s); checker success means
  the **expected failure** was reproduced, never a successful RX compile.
- Initial device-free GREEN: all24 cases return0,13.174938844s serial compile
  total, input roots pinned, no device mappings/discovery. A successor check
  also retains dependency hashes, fixture-inventory validation and export probe.
- Successor device-free GREEN: all24 return0,13.482626127s total,2702 dependency
  hashes retained and attempt-owned generated/fabric write probe passed.
- Four isolated new-layout gates passed: A/B exact selectors list with eager
  loading, and A/B no-chip probes each exit1/No chips detected in the cluster;
  no device mappings, all containers cleaned. [compile-proof.json](compile-proof.json)
  retains exact argv/results. Its typed publication keeps each dependency as
  explicit path/SHA256 records (filename-keyed digests triggered a token-name
  secret false positive); initial JSON is preserved byte-exact in
  compile-proof-initial.tar.gz, all paths/hashes/measurement values unchanged.
  No analysis finding was accepted/muted. A successor proof/receipt-v9 binds
  the clarified schema/processor guard and unchanged native layout/ELFs; v8
  remains historical at its revision. [compile-proof-v3.json](compile-proof-v3.json)
  repeats current controls: old RX include error reproduced,24 GREEN cases0,
  13.435495187s/2702 dependency entries/export write passed. Four native gates
  are reused byte-exact after runtime/command/transport AST equality against v8.
  Full-suite/ruff/CI/Sonar are in the report.

No new A/B is run by this preparation. [morning-admission.json](morning-admission.json)
keeps B physically gated pending new owner approval and fresh-A handling after
card power-off; host boot UUID alone cannot make the old link evidence current.
See the precommitted Bv2 prediction and combined morning proposal in the
[first-transfer plan](../issue30-first-transfer-plan.md).
