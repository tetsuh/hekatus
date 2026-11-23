"""Pinned custom-runner entry point; execution remains explicitly gated."""
from enodia.tt.bench.issue120_production.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
