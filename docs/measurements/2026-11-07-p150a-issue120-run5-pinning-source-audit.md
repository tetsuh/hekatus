## Board-free source assessment after the P5/P6 execution stopped

**Yes, conditionally:** when passthrough results in the target PCI device's effective IOMMU group type being `identity`, this audited pinning capability query returns **`{max_pins=0, max_total_pin_size=0, can_map_to_noc=false}`**. The literal kernel command-line token `iommu=pt` is not itself read by UMD, so the token alone is not an unconditional proof of zero for every device.

### Audited versions and exact chain

The staged source files were byte-compared to TT-Metal v0.75.0 commit `d9a68815f5fcf08a5bfbffb6f1f811823fba8edd` and its gitlinked UMD commit `bc77ac2f7382ffeb9fa5430055f0854d53979489` (not the unrelated current UMD checkout). All three comparisons matched.

1. [`GetMemoryPinningParameters`, lines 388-402](https://github.com/tenstorrent/tt-metal/blob/d9a68815f5fcf08a5bfbffb6f1f811823fba8edd/tt_metal/distributed/pinned_memory.cpp#L388-L402) reads Metal cluster `is_iommu_enabled()`. If false, lines 391-392 immediately return `{0u, 0u, false}` **before HAL pin limits or 64-bit PCIe/NoC support are consulted**. Blackhole's addressing support does not bypass that early return.
2. [Metal cluster cache, lines 335-348](https://github.com/tenstorrent/tt-metal/blob/d9a68815f5fcf08a5bfbffb6f1f811823fba8edd/tt_metal/llrt/tt_cluster.cpp#L335-L348) defaults the flag to false, then for Silicon with an MMIO target/PCI object takes the first selected MMIO PCI device's `is_iommu_enabled()`.
3. [UMD `detect_iommu`, lines 100-106](https://github.com/tenstorrent/tt-umd/blob/bc77ac2f7382ffeb9fa5430055f0854d53979489/device/pcie/pci_device.cpp#L100-L106) reads the device-relative sysfs attribute `iommu_group/type`. It returns true **only if the first three characters are `DMA`** (the comment names `DMA`/`DMA-FQ`); otherwise false. The [PCI constructor, lines 386-395](https://github.com/tenstorrent/tt-umd/blob/bc77ac2f7382ffeb9fa5430055f0854d53979489/device/pcie/pci_device.cpp#L386-L395) caches that result.

Thus the explicit zero-tuple branch is reached for:

- Effective group type `identity` (passthrough), or any readable type not starting with `DMA`.
- Missing/unreadable/unparseable group-type sysfs input, for which `try_read_sysfs` supplies no value and detection returns false.
- Metal cache still false for a non-Silicon/no-MMIO/no-PCI-object path.

For a device whose effective group remains `DMA`/`DMA-FQ` despite a passthrough global default, this early-zero branch is **not** taken; HAL supplies the limits, which still need separate assessment. The code does not equate a boot option, IOMMU hardware presence, translated DMA protection and available pinning capacity.

### Why passthrough can yield `identity`

As a public kernel reference, Linux v6.12 [x86 `iommu_setup`](https://github.com/torvalds/linux/blob/v6.12/arch/x86/kernel/pci-dma.c) handles `pt` with `iommu_set_default_passthrough(true)`. [IOMMU core](https://github.com/torvalds/linux/blob/v6.12/drivers/iommu/iommu.c) sets the default to `IOMMU_DOMAIN_IDENTITY` and exposes that domain as `identity`. Its per-device/default-domain selection can override/fall back, so check the effective group, not only the global token. The [sysfs ABI](https://github.com/torvalds/linux/blob/v6.12/Documentation/ABI/testing/sysfs-kernel-iommu_groups) distinguishes translated `DMA`/`DMA-FQ` from untranslated `identity`.

### Limits and no host changes

This is source evidence, not a new native capability-query run. P4's retained stdout reports `IOMMU: disabled`, consistent with the early-zero path, but P4 did not record the actual returned tuple/group type. Full image/source parity is still not asserted. Linux v6.12 is an audit reference, not an assertion about the running host kernel. I did not read or modify the current kernel command line/group setting, change host configuration, rebind devices, run a capability query, reset or retry P4. This assessment authorizes no such action.
