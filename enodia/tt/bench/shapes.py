"""The matmul shapes this workload actually produces, with FLOP accounting.

Separated from the runner on purpose. A benchmark's honesty lives in two
places — which shapes it measures, and how it converts a duration into a
FLOP count — and neither should require an accelerator to review. This
module imports nothing beyond the standard library, so it is testable
anywhere.

**Why not a large square matmul.** That is the shape an accelerator is best
at, and it is not the shape here. The MV inverse is a batch of small complex
matmuls (design.md §9); beamspace projection is wide and very thin; the
front-end FIR has a narrow output. Quoting a square-matmul figure as this
workload's efficiency would overstate it, so the catalogue carries one such
shape explicitly labelled as a reference, and the gap between it and the
representative shapes is measured rather than argued.

**Complex arithmetic.** A complex matmul is counted as four real matmuls,
the naive decomposition. Karatsuba would make it three at the cost of extra
additions; if the implementation adopts it, the count here changes with it,
because this number is the denominator of every efficiency figure.
"""

from __future__ import annotations

from dataclasses import dataclass

# Front-end complex band-pass FIR, design.md §5.
FIR_TAPS = 64

# MV subaperture sizes, design.md §10: 32 for a 64-channel receive, 64 for
# 128 channels, and 16 as the beamspace dimension that makes 2D tractable.
SUBAPERTURE_SIZES = (16, 32, 64)

# Receive channel counts the design works with, design.md §3.
CHANNEL_COUNTS = (128, 256)

# The beamspace dimension the design projects onto, design.md §9.
BEAMSPACE_DIM = 16


@dataclass(frozen=True)
class MatmulShape:
    """One matmul as the workload issues it.

    `real_matmuls` is how many real matmuls one logical operation costs, so
    a complex operation is 4 and a real one is 1. It is stated per shape
    rather than derived from a flag, because the front-end FIR is neither:
    real samples against complex coefficients cost two.
    """

    name: str
    batch: int
    m: int
    k: int
    n: int
    real_matmuls: int
    family: str
    note: str

    def __post_init__(self) -> None:
        for field, value in (
            ("batch", self.batch),
            ("m", self.m),
            ("k", self.k),
            ("n", self.n),
            ("real_matmuls", self.real_matmuls),
        ):
            if value < 1:
                raise ValueError(f"{self.name}: {field} must be at least 1, got {value}")

    @property
    def representative(self) -> bool:
        """False for shapes present only as a flattering contrast."""
        return self.family != "reference"


def total_flops(shape: MatmulShape) -> int:
    """Real floating-point operations for one execution of the shape.

    2*M*K*N per real matmul — one multiply and one add per accumulation.
    """
    return shape.batch * shape.real_matmuls * 2 * shape.m * shape.k * shape.n


def newton_schulz_shapes(
    batches: tuple[int, ...] = (1024, 8192, 65536),
) -> list[MatmulShape]:
    """One Newton-Schulz matmul: a batch of small square complex matmuls.

    The iteration X <- X (2I - R X) is two complex matmuls per iteration,
    fixed at 8 iterations (design.md §9), over one covariance matrix per
    pixel. The batch dimension is what keeps the matrix engine busy, and it
    is the reason this beats a sequential solver despite paying far more
    FLOPs. The batch sizes bracket a frame: a 13 MHz frame is on the order
    of 10^5 pixels.
    """
    return [
        MatmulShape(
            name=f"newton_schulz_L{size}_b{batch}",
            batch=batch,
            m=size,
            k=size,
            n=size,
            real_matmuls=4,
            family="newton_schulz",
            note=f"MV inverse step, subaperture L={size}",
        )
        for size in SUBAPERTURE_SIZES
        for batch in batches
    ]


def beamspace_shapes(
    pixel_counts: tuple[int, ...] = (4096, 65536),
) -> list[MatmulShape]:
    """Projection of channel space onto the beam basis.

    A wide, very thin output: B rows against 128 or 256 channels. This is
    what makes MV tractable on a 2D probe at all (design.md §9), so its
    efficiency matters out of proportion to its FLOP count.
    """
    return [
        MatmulShape(
            name=f"beamspace_B{BEAMSPACE_DIM}_ch{channels}_p{pixels}",
            batch=1,
            m=BEAMSPACE_DIM,
            k=channels,
            n=pixels,
            real_matmuls=4,
            family="beamspace",
            note=f"channel space {channels} projected onto {BEAMSPACE_DIM} beams",
        )
        for channels in CHANNEL_COUNTS
        for pixels in pixel_counts
    ]


def frontend_fir_shapes(
    output_widths: tuple[int, ...] = (2, 8, 32),
    samples: int = 262144,
) -> list[MatmulShape]:
    """The front-end FIR expressed as a matmul.

    K is the tap count and the output is narrow, which is a poor matmul
    shape however cheap it is in FLOPs. The widths sweep that directly: 2 is
    one complex output, and the wider variants stand for computing several
    decimation phases in one operation. If efficiency climbs with width, the
    front end should be organized to produce several phases at once.
    """
    return [
        MatmulShape(
            name=f"frontend_fir_taps{FIR_TAPS}_w{width}",
            batch=1,
            m=samples,
            k=FIR_TAPS,
            n=width,
            real_matmuls=2,
            family="frontend_fir",
            note=f"{FIR_TAPS}-tap complex FIR, {width}-wide output",
        )
        for width in output_widths
    ]


def reference_shapes(sizes: tuple[int, ...] = (4096,)) -> list[MatmulShape]:
    """Large square matmuls, included as contrast and never as a result.

    This is the number a generic benchmark would report. Measuring it in the
    same run is what turns "generic benchmarks flatter" from an assertion
    into a measured gap.
    """
    return [
        MatmulShape(
            name=f"reference_square_{size}",
            batch=1,
            m=size,
            k=size,
            n=size,
            real_matmuls=1,
            family="reference",
            note="flattering shape, for contrast only",
        )
        for size in sizes
    ]


def default_catalogue() -> list[MatmulShape]:
    """Every shape, representative ones first."""
    return [
        *newton_schulz_shapes(),
        *beamspace_shapes(),
        *frontend_fir_shapes(),
        *reference_shapes(),
    ]
