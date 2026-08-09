# hekatus

> A real-time ultrasound compute stack.
>
> Named for Hekate — the goddess who alone heard Persephone's cry, and who
> carried torches into the dark to search for her — by way of *Hekatos*,
> "the one who works from afar and strikes the mark at will."
> The tail of the name is for ultrasound.
>
> **enodia** walks the path, **diaplous** carries the image across,
> **lampas** burns beside them.

## What this is

hekatus is the compute side of a medical ultrasound imaging system:
everything that happens after received channel data reaches the wire.
Transmit control, transmit hardware, and the operator UI live outside.

| Role | Greek | Meaning | Scope |
|---|---|---|---|
| **enodia** | Ἐνοδία | *on the road* — an epithet of Hekate | receive, front-end, delays, retrospective transmit focusing, beamformers, envelope — the deterministic real-time path |
| **diaplous** | διάπλους | *the crossing* | mode-specific back-end processing and scan conversion — carrying data across from acoustic geometry to the displayed image |
| **lampas** | λαμπάς | *the torch* | companion inference, observing the pipeline non-intrusively |

Roles are hardware-neutral concepts; placing a role onto particular hardware
is a separate, changeable decision. The current proof-of-concept places
enodia on a Tenstorrent accelerator. The reference implementation is the
hardware-neutral specification: any port is accepted by numerical equivalence
against it, not by code reuse.

## Status

Bootstrapping. The design documents are being prepared for publication
(translation and review) — see the open issues and milestones.

- Development process: [docs/development_workflow.md](docs/development_workflow.md)
  (entry point for coding agents: [AGENTS.md](AGENTS.md))
- Decisions and their evolution: [docs/adr/](docs/adr/)

## License

[Apache-2.0](LICENSE)
