"""
OutputManager — enforces canonical output layout for research experiments.

Directory contract (relative to research/experiments/):

    experiments/outputs/
    ├── scratch/    — hot, raw, iterative data (trade logs, temp CSVs, SQLite)
    └── artifacts/  — finalized outputs (summary CSVs, equity-curve PNGs)

Usage:
    from research.engines.output_manager import OutputManager

    om = OutputManager()
    trade_log_path  = om.scratch_path("trades_1810hk.csv")
    chart_path      = om.artifact_path("tournament_20260517.png")
    summary_path    = om.artifact_path("comparison_report.csv")
"""

from __future__ import annotations

from pathlib import Path


class OutputManager:
    """
    Resolve and create the two-tier output directory structure.

    Scratch  — ephemeral, local-only, not synced to production:
               raw trade logs, iterative CSVs, temporary SQLite snapshots.
    Artifacts — promoted outputs ready for review or DB sync:
                final summary CSVs, equity-curve PNGs.
    """

    _OUTPUTS_ROOT: Path = (
        Path(__file__).resolve().parent.parent / "experiments" / "outputs"
    )

    def __init__(self) -> None:
        self.scratch: Path = self._OUTPUTS_ROOT / "scratch"
        self.artifacts: Path = self._OUTPUTS_ROOT / "artifacts"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)

    def scratch_path(self, filename: str) -> Path:
        """Return a Path inside scratch/. Does not create the file."""
        return self.scratch / filename

    def artifact_path(self, filename: str) -> Path:
        """Return a Path inside artifacts/. Does not create the file."""
        return self.artifacts / filename
