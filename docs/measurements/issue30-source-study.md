# Issue #30: pinned Ethernet source study (board-free)

## Invariants and evidence boundary

- No board access, device open, board run, reset, flashing or `tt-smi` invocation
  was performed. The existing development checkout of tt-metal was not used.
- Preserve the absolute rules: downstream never makes enodia wait; finite
  buffering needs capacity/violation detection, not downstream synchronization;
  data-plane tags and parameter-plane distribution remain separate.
- A host API reaching discovery is not proof that it starts PHY training. A
  trained link is not proof of payload delivery. A TT peer working is not proof
  of FPGA interoperability. Necessary conditions below are not sufficient ones.
- This is source evidence, not a receive-path architecture selection or timing
  measurement. No new kernel or production protocol is introduced.

## Reproduction and reference convention

A separate shallow checkout was obtained under ignored `out/issue30/tt-metal`:

```bash
git clone --depth 1 --branch v0.75.0 https://github.com/tenstorrent/tt-metal.git out/issue30/tt-metal
git -C out/issue30/tt-metal rev-parse HEAD
git -C out/issue30/tt-metal submodule update --init --depth 1 tt_metal/third_party/umd
git -C out/issue30/tt-metal submodule status
```

| Evidence | Pin |
|---|---|
| **T**, tt-metal v0.75.0 | `d9a68815f5fcf08a5bfbffb6f1f811823fba8edd` |
| **U**, its UMD gitlink (not the host Python UMD package) | `bc77ac2f7382ffeb9fa5430055f0854d53979489` |
| Cluster-descriptor gitlink (not initialized/used as hardware truth) | `7b2176e2fe913089f8cd2be9dfb738ead6e7aa27` |
| Runtime image | `ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621` |
| **P**, supplementary TT-Topology README, not part of T | `7bd675f888f9f0d27674fab7cb4188d5b1255481` |

`T:path:line` is relative to the T checkout; `U:path:line` to its UMD checkout.
These are immutable source references, not local machine paths. Browse roots:
[T](https://github.com/tenstorrent/tt-metal/tree/d9a68815f5fcf08a5bfbffb6f1f811823fba8edd),
[U](https://github.com/tenstorrent/tt-umd/tree/bc77ac2f7382ffeb9fa5430055f0854d53979489),
[P](https://github.com/tenstorrent/tt-topology/blob/7bd675f888f9f0d27674fab7cb4188d5b1255481/README.md).
Line ranges are inclusive. Firmware is **not a submodule** in T's `.gitmodules:1-16`
or U's git tree. U includes firmware ABI headers; this does not pin the source
of the installed CMFW/base ERISC implementation. Its source revision remains
**open**. Runtime-compiled application firmware is in T, not a new flash image.

The image was inspected with a named **never-started**, network-disabled Docker
container, export and copy only, then removed. No device was mapped. Its wheel
metadata names ttnn 0.75.0. Two copied source files match T byte-for-byte:

| T path | SHA-256 (also equal in the wheel) |
|---|---|
| `tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp` | `3d109db29454c4feb775debdc86a0589d30c075cb39352dc7d17821c97651fc6` |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | `723fb3a55c92126b258d4c5b766f7d2bb00e9f39a08761944b4c4df0994191a9` |

This is a two-file packaging check, **not** proof of every linked library's
source identity. The exported image has neither `fabric_smoke_tests` nor
`fabric_unit_tests`; building an official test is a prerequisite of the plan.

## 1. Which datamover is current?

**Conclusion:** TT-Fabric's persistent EDM/router is the current fabric path;
`handshake::deprecated` belongs to the retained **non-persistent legacy CCL**
EDM. They coexist; the namespace is not a description of every Ethernet API.
There is no source basis for saying every MeshDevice enables fabric by default,
or that fabric is a supported generic external Ethernet receiver.

| Layer / caller | What the pin actually does |
|---|---|
| Legacy EDM | `T:ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp:146-151,226-232` calls the split deprecated handshake. The retained builder still names this kernel at `T:ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:537-610`. Presence does not establish use by all current CCL operations. |
| Explicit deprecation | `T:tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp:124-132` says these split APIs were developed for non-persistent EDM kernels, not supported with TT-Fabric, and are to be removed with legacy CCL ops. This is the source's migration direction, not an owner ingest-path decision. |
| Fabric EDM | `T:tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:53-70,3518-3603` implements packet forwarding/local NoC delivery and uses the fabric handshake, not the deprecated split handshake. Host construction is `T:tt_metal/fabric/compute_mesh_router_builder.cpp:397-419`. |
| MeshDevice | `T:tt_metal/distributed/mesh_device.cpp:343-431,493-542` maps fabric node IDs and initializes fabric/dispatch firmware. However `T:tt_metal/impl/context/metal_env_impl.hpp:110` starts with `FabricConfig::DISABLED`. Node IDs/control-plane metadata do not themselves imply traffic on Ethernet. |
| Dispatch exception | `T:tt_metal/impl/device/device_manager.cpp:225-242,270-299` may open the whole cluster and auto-enable `FABRIC_1D` for remote-chip dispatch if disabled. Two PCIe-capable p150a chips do not become remote chips merely because they are cabled together. Explicit fabric selection is appropriate for the transfer test. |
| Representative current ttnn CCL | `T:ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp:68-89` selects composite/primitive all-gather; its program factory calls the shared async builder at `T:ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_program_factory.cpp:108-139`. The shared builder wires fabric connections at `T:ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp:760-786`; its writer opens them at `T:ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp:202-226`. The fabric configuration helpers accept 1D/2D and throw for unsupported configs at `T:ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:1743-1777,1799-1845`. This is a traced example, not an audit of every ttnn operation. |
| Sockets | `T:ttnn/api/ttnn/distributed/fabric_socket.hpp:13-22` documents FabricSocket wrapping MeshSocket. `T:tt_metal/distributed/mesh_socket_utils.cpp:56-100` requires enabled fabric for cross-device connections; disabled fabric is allowed for same-device endpoints. This is not a host TCP socket or a PHY-training API. |
| Low-level Ethernet | `T:tt_metal/hw/inc/internal/ethernet/tunneling.h:82-119` issues remote-L1 transfer commands. This primitive exists below both application protocols; sharing it does not make the protocols interchangeable. |

Thus design section 3's earlier channel state machine describes the legacy
implementation, not the persistent fabric router. In particular the legacy
receiver signals workers to consume its buffers (`erisc_async_datamover.hpp`,
next section); it is not evidence that that kernel pushes every received byte
directly into a worker's L1.

## 2. Link training: what can actually be called?

### Host entry and the limit of the source claim

The minimal official fabric fixture calls `SetFabricConfig(FABRIC_1D)` followed
by `MeshDevice::create_unit_meshes(ids, ...)` for **all available chips**:
`T:tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:79-127,196-200`.
That reaches runtime initialization/discovery; it is the candidate transfer
entry, **not a demonstrated link-training command**.

The UMD chain is explicit:

```text
Cluster construction -> TopologyDiscovery::discover
  -> get_connected_devices -> TTDevice::init_tt_device
  -> wait_eth_cores_training (default enabled)
  -> retrain_eth_cores [Blackhole: skipped]
  -> discover connected peers -> ClusterDescriptor
Metal fabric setup -> logical mapping/routing -> application EDM handshake
  -> packets -> receiver content verification
```

Evidence:

- `U:device/cluster.cpp:350-366`: silicon construction uses topology discovery.
- `U:device/topology/topology_discovery.cpp:112-138,171-220,606-620`:
  initialization, training wait, then discovery.
- `U:device/api/umd/device/topology/topology_discovery_options.hpp:78`: wait
  defaults to true; `U:device/api/umd/device/utils/timeouts.hpp:32`: default
  training timeout is **900,000 ms**, not this project's 60-second cap.
- `U:device/tt_device/tt_device.cpp:137-153`: init probes ARC, waits for ARC,
  creates telemetry/firmware providers and the SoC descriptor. It does not
  contain a Blackhole PHY-training-start call.
- `U:device/tt_device/blackhole_tt_device.cpp:309-336`: wait polls the firmware
  `port_status` until not IN_PROGRESS. Its timeout logs and breaks rather than
  throwing; its returned `time_taken` stays zero. Consequently neither a
  completion log nor successful host return alone proves PORT_UP.
- `U:device/topology/topology_discovery_blackhole.cpp:207-209` explicitly skips
  host retraining. `U:device/topology/topology_discovery.cpp:730-731` includes
  links as trained only on SUCCESS.

**Important correction to the previous runtime-training shorthand:** the pin
exposes firmware status and runtime cooperation with firmware; this study
finds no standalone host call guaranteed to initiate first training on two
never-linked p150a cards. It cannot establish that opening a MeshDevice will
turn their down ports up. First-run success depends on already functioning
base firmware and physical links. If discovery finds none, stop there rather
than try fabricated descriptors, direct register pokes or repeated opens.

### Base firmware versus application firmware

Blackhole runtime's `aerisc_context_switch()` services firmware messages and
updates link status, only on ERISC0:
`T:tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h:331-371`.
`update_boot_results_eth_link_status_check()` calls a function pointer in the
**base firmware ABI** after a timestamp debounce (`:338-350`). The interval
constant is 1000 ms (`:32-33,256-264`), not a measured duration of that call.
The implementation behind that pointer, its first-training trigger and its
recovery behavior on the installed firmware are **open** here.
`T:tt_metal/hw/inc/internal/ethernet/erisc.h:20-58` distinguishes the full
context switch (message service, NoC sync/counter handling) from the no-sync
path (Blackhole link-status check only). Do not conflate them.

Firmware prerequisites are layered, not one invented version requirement:

| Source / existing record | What it establishes, and what it does not |
|---|---|
| `U:device/firmware/firmware_info_provider.cpp:324-337` | Blackhole minimum compatible **bundle** 18.5.0; UMD latest-supported constant 19.7.1. These are not proof of first-link behavior or an exact CMFW requirement. |
| `U:device/api/umd/device/firmware/erisc_firmware.hpp:20-25` and `U:device/topology/topology_discovery.cpp:628-664` | Minimum ERISC 1.4.1 and per-core version consistency checks; older expected versions generate a warning. |
| `U:device/topology/topology_discovery.cpp:549-600,667-727` | Bundle consistency/support and firmware heartbeat checks; a flash-bundle label alone is not an ERISC health check. |
| `T:INSTALLING.md:26-42` | Installation guidance lists Blackhole bundle 19.8.1 and KMD >=2.8.0. This differs from UMD's latest-supported constant. Record the discrepancy; do not silently upgrade. |
| Existing `2026-11-14-p150a-issue120-p6-api-lock-semaphore-v3-env.json` | Historical device0 bundle 19.6.0.0 / CMFW 0.28.0.0, telemetry eth_fw 0.0.0. Not a fresh two-card check, nor the required base ERISC version. |

No firmware source submodule revision can be supplied where none exists.
Owner must confirm actual versions on **both** cards; an update or a baseline
change is a separately approved intervention, not part of this study.

### Descriptors and TT-Topology

- The discovered **cluster descriptor** reports physical chips and Ethernet
  peers. Firmware peer-board ID and channel reads are in
  `U:device/topology/topology_discovery_blackhole.cpp:51-64,103-145`.
  A user-authored descriptor does not train a physical link.
- The **mesh graph descriptor (MGD)** expresses logical workload topology,
  not a training script (`T:tt_metal/fabric/MGD_README.md:3-26`). Two P150 chips
  classify as P150_X2 (`T:tt_metal/llrt/tt_cluster.cpp:182-190`); default MGD
  selection is `T:tt_metal/fabric/mesh_graph.cpp:72-85`.
  `T:tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto:3-12`
  describes one host, a 1x2 mesh, **4 channels with RELAXED policy**. Do not
  infer four trained channels, a particular panel port, or successful mapping.
- No custom MGD is the first candidate. If discovered connectivity cannot
  satisfy mapping, stop and preserve the descriptor/error before proposing
  changed configuration.
- `P:README.md:3-16` explicitly excludes BH PCIe cards; `:112-118` describes
  flashing Ethernet coordinates and board resets. That tool is neither
  supported for p150a nor needed to construct the runtime MGD. In T,
  `INSTALLING.md:203-207` mentions it only for some multi-card systems, not a
  Blackhole enablement procedure. No invocation is proposed.

### Where to observe status (not observed in this study)

| Observation | Meaning / caveat |
|---|---|
| UMD `TTDevice::read_eth_core_training_status` | Reads `BOOT_RESULTS_ADDR + offsetof(eth_status_t, port_status)`; `U:device/tt_device/blackhole_tt_device.cpp:331-336`. `U:device/api/umd/device/types/blackhole_eth.hpp:37-44,180-184` defines port states and asserts port-status address `0x7CC04`. Use the API and correct coordinate system, not guessed raw writes. |
| Firmware heartbeat / postcode / peer | `U:device/topology/topology_discovery_blackhole.cpp:188-205` and the peer fields above help distinguish base-firmware readiness from no peer. |
| Discovered cluster connections | `T:tt_metal/llrt/tt_cluster.hpp:312-314` and `tt_cluster.cpp:1379-1384` expose descriptor-derived link state, **not a continually sampled PHY monitor**. Refresh/reopen is not authorized by reading this document. |
| Official health report | `T:tests/tt_metal/tt_fabric/system_health/test_system_health.cpp:257-340` reports link/peer/retrain information. The source explicitly says this test does not fail despite reported errors. Zero exit is not a health verdict. Blackhole CRC/codeword reads are excluded by a Wormhole check (`:315-324`); displayed zeros are not measured Blackhole error counts. |
| `ETH_LIVE_STATUS`, `ETH_FW_VERSION` in saved tt-smi telemetry | `U:device/firmware/firmware_info_provider.cpp:212-230` says BH ETH_LIVE_STATUS is always zero before 19.9 and ETH_FW_VERSION is unimplemented. On >=19.9, upper 16 bits are link, lower 16 heartbeat. Thus the historical 19.6 zeros cannot establish link-down or ERISC version. Host tt-smi's formatting/version is separate from pinned UMD; fresh field availability remains unverified. |

Panel-port-to-ETH-channel mapping is **open**: the official health test labels
P150 channel values 4..11 as QSFP (`system_health/test_system_health.cpp:85-91`), but
that does not specify which physical connector each pair uses. Physical cable
tracing and an authoritative board port map must precede inference from channel
numbers. Do not transplant Wormhole port diagrams.

## 3. What the peer must imitate

### Legacy protocol (explains the old design reading, not a fabric recipe)

`T:tt_metal/hw/inc/internal/ethernet/tunneling.h:31-60` defines the 16-byte
`eth_channel_sync_t`: `bytes_sent`, `receiver_ack`, `src_id`, `reserved_2`.
`erisc_info_t` has four launch/unused words followed by the channel array.
Addresses and channel/buffer counts are configured; the struct alone is not a
wire specification.

`T:tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp:145-203` initializes
handshake+16 words to `[1,1,0x1c0ffee1,0x1c0ffee2]`, clears channel state,
sends a 16-byte handshake on channel 0 and waits for receiver completion.
The receiver must finish channel initialization before completing the handshake.

`T:ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_async_datamover.hpp:232-280,499-587`
implements two distinct acknowledgements:

1. Nonzero `bytes_sent` indicates a received buffer. A separate 16-byte first
   ack with `receiver_ack=1` lets the sender reuse its **local source** buffer.
2. Workers are signalled with NoC semaphores to consume the receive buffer.
   Only when worker completion is seen does the receiver return zeroed
   `bytes_sent`/`receiver_ack`, allowing reuse of the **remote destination**.

The ack sources must not alias: the source explicitly documents the race
between first and second ack (`:249-258`). Multi-buffering advances indices;
copying only the state enum or only the ack flag cannot reproduce the protocol.

### Fabric protocol (candidate current path)

```text
worker packet + header -> local EDM sender slots
    -> remote EDM receiver slots -> local NoC destination and/or next router
                 <- first ack (when enabled) / completion credits
```

- **Initialization handshake:**
  `T:tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp:33-117` defines
  `MAGIC_HANDSHAKE_VALUE=0xAA` and a 32-byte `handshake_info_t`. Bytes 0..3
  hold `local_value`, 4..5 neighbor mesh ID, 6 neighbor device ID, 7 padding,
  8..15 padding, 16..31 `scratch[4]`. Init zeroes local_value and puts magic
  and packed local mesh/device identity in scratch[0..1]. The designated
  sender repeatedly copies its scratch's first 16 bytes to the peer's first
  16 bytes until the reverse reply arrives; receiver replies after seeing magic.
  `fabric_router_eth_handshake.hpp:21-90` adds termination checks. The actual
  router calls these at `fabric_erisc_router.cpp:3518-3560`; there are further
  local-ERISC coordination/status phases at `:3562-3603`. This is not the
  channel-0 split handshake above.
- **Buffers/credits:**
  `T:tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp:196-259`
  tracks remote slot addresses, slot sizes, free slots, send/flush/ack/completion
  counters and source channel IDs. `fabric_router_flow_control.hpp:10-80,119-177,239-257`
  implements counter-memory or stream-register credit return, selected at
  compile time. First-level receipt is distinct from completion. The receive
  loop checks NoC transaction flush before returning completion credits:
  `fabric_erisc_router.cpp:1928-1976`. The peer must agree on the **actual
  generated** layout, credit mode, counts, wrap behavior and stream IDs;
  hardcoding the legacy `erisc_info` layout is insufficient.
- **Header:** `T:tt_metal/fabric/fabric_edm_packet_header.hpp:378-420` has
  40-byte NoC command union, uint16 payload bytes, NoC send type and source
  channel ID. NoC unicast's command contains the destination address
  (`:164-186,448-462`). Routing fields differ by header type: the 1D
  `LowLatencyPacketHeaderT` has 48-byte or 64-byte total layouts depending
  on extension words (`:1024-1060`); other packet types are not interchangeable.
  This is **not** the deprecated per-channel shared struct. Workers build
  headers and the EDM sets/updates routing/channel fields
  (`fabric_erisc_router.cpp:523-553`).
- **Transport primitive:** `T:tt_metal/hw/inc/internal/ethernet/tunneling.h:82-136`
  programs TX queue destination L1 addresses/transfer sizes and remote-register
  writes. `fabric_router_flow_control.hpp:58-73` returns credits through such
  remote stream-register updates. An FPGA MAC producing payload bytes alone
  has not implemented this remote-memory/register contract.

**External-source boundary:** slot discipline, content verification and the
application's chosen payload/header semantics can be described independently
of who generates data. However PHY/link bring-up, remote L1 addressing,
remote-register writes, firmware peer discovery/identity, reverse credits,
ERISC cooperation and routing-table setup in this implementation assume a TT
peer. No supported FPGA personality, link-training sequence, complete on-wire
framing/FEC configuration or compatibility guarantee is established by these
sources. An external sender would need to implement that contract or use a
separately designed receive adaptation; neither is selected here. The source
lists software obligations, **not a sufficient FPGA implementation spec**.
Credit backpressure here also does not authorize downstream blocking of enodia.

## 4. B3: can a routing/context-switch call occur during a continuous stream?

**Yes, it can; continuous arrival is not a sufficient exclusion condition.**
No jitter duration has been measured.

| Path | Exact source condition |
|---|---|
| Legacy main loop | `T:ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp:237-238,248-324`: reset idle count on a sender/receiver state-step reporting progress; otherwise `did_nothing_count++ > 4000000` calls `run_routing()`. Starting at zero this is 4,000,002 unsuccessful outer iterations, not 4 million cycles. Successful progress often enough prevents **this branch only**. A busy TX queue, credit wait or worker completion wait can prevent progress despite an externally continuous source. Teardown also routes after 5,000,000 waits (`:336-356`), and handshake/wait helpers can context-switch. |
| Fabric defaults / actual BH override | `T:tt_metal/fabric/erisc_datamover_builder.hpp:449-450,642-643`: builder default 10,000 / WAIT_FOR_IDLE. `compute_mesh_router_builder.cpp:413-419`: BH with default-enabled two-ERISC mode overrides to **32 / WAIT_FOR_IDLE**; `T:tt_metal/llrt/rtoptions.hpp:294` has the default. Do not carry the legacy 4M number into fabric. |
| Fabric traffic progress | `fabric_erisc_router.cpp:2272-2276` clears `did_something` per outer loop. Sender sets it when it can send (`:1655-1669`); receiver sets it when forwarding is possible (`:1889-1891`). This is local progress, not physical line activity. Credit/NoC/downstream stalls can leave it false. |
| Fabric idle vs periodic | `fabric_erisc_router.cpp:2595-2614`: with enable_context_switch and IDLE_CONTEXT_SWITCHING, progress resets the counter; otherwise `counter++ > SWITCH_INTERVAL` enters the coordinated no-sync path. With INTERVAL it does so regardless of progress. Thus interval 32 is 34 unsuccessful checked outer loops from zero, not 34 packets or cycles. Dispatch router separately chooses 16 / INTERVAL (`compute_mesh_router_builder.cpp:840-842`). |
| Two-ERISC coordination | `erisc_datamover_builder.cpp:119-144`: ERISC0 TX is master/context-switch enabled; ERISC1 RX is subordinate. `fabric_erisc_router.cpp:1400-1490` makes the subordinate stop for the master's request. **Inference:** receive-only continuous traffic can coexist with an idle local sender/master and hence a coordinated pause of the busy receiver. Global activity cannot prove the master resets its counter. |
| Work actually invoked | `T:tt_metal/hw/inc/internal/ethernet/tunneling.h:178-186`, `internal/ethernet/erisc.h:20-58`, `internal/tt-1xx/blackhole/eth_fw_api.h:338-350`: legacy `run_routing` uses the full path; fabric's main path uses no-sync link-status checking on BH. The timestamp debounce can skip the base-firmware call; when due, opaque firmware work remains. No timing bound follows from the idle count or the 1000-ms debounce constant. |

**Later measurement proposal, not an execution instruction:** after one verified
transfer, separately preregister bounded finite packet trains and controlled
gaps, with card-side receive-completion intervals, packet sequence/checksum,
drop/overflow counts and before/after context-switch timestamps/counters.
Compare the actual default generated router configuration with its recorded
progress, rather than assuming that offered load is progress. First characterize
instrument overhead with Watcher, then a separately approved Watcher-off pair;
log entry/exit of the relevant helper if attribution is needed (new kernel =>
first run Watcher on). Record median/tails/max and conditional distributions
near calls; do not infer routing causality from a tail alone. First smoke's
five packets cannot estimate P99.9. Cross-card one-way timing requires B6
calibration; host visibility ordering still needs B5. Same-card intervals or a
round trip avoid claiming synchronized clocks but are not calibrated one-way
latency. #105 stage 3 remains blocked and needs its own AC before starting.

## Open ledger and next gate

1. **First-training trigger:** installed CMFW/base ERISC source revision,
   initial boot/training sequence and exact runtime action that can bring up
   these never-linked ports. Discovery waits; Blackhole host retrain is skipped.
2. **Physical configuration:** connector pairs, cable properties and actual
   channel/port/harvesting mapping on both cards; no observed link here.
3. **Firmware compatibility:** reconcile installation guidance, UMD constants
   and the fresh owner-approved two-card baseline, without an automatic update.
4. **First transfer:** official smoke binary must be built/provenanced; it is
   not shipped in the inspected runtime image. Then mapping, handshake, content
   check, teardown and <=60-second feasibility must be observed, not presumed.
5. **FPGA interoperability and production receive architecture:** not settled
   by chip-to-chip source or by a future successful TT-to-TT smoke.
6. **B3/B4 cost:** conditions are source-resolved; jitter and card-to-card timing
   remain unmeasured. B5/B6 remain open. Neither #13 nor #105 is unblocked by
   this document alone.

The [first-transfer plan](issue30-first-transfer-plan.md) deliberately allows a
recorded stop at discovery/training. This completes only the source-study and
plan portion of #30, not the board-transfer acceptance criterion.
