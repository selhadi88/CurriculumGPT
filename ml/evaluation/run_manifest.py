"""
Run manifest — captures the full provenance of a research run so results are
reproducible and auditable.

Every pipeline run writes a timestamped folder under `outputs/runs/<run_id>/`
containing: manifest.json (config + environment), all metric JSONs, the trained
checkpoint path, and the generated report. This is the canonical research record.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_OUTPUTS_ROOT = Path(__file__).resolve().parents[2] / "outputs"
RUNS_ROOT = _OUTPUTS_ROOT / "runs"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["device"] = (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu")
    except Exception:  # noqa: BLE001
        env["torch"] = "n/a"
    return env


class RunManifest:
    """Create and populate a single research run's output folder."""

    def __init__(self, run_id: str | None = None, config: dict | None = None) -> None:
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir = RUNS_ROOT / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": self.run_id,
            "config": config or {},
            "environment": _environment(),
            "artifacts": {},
            "metrics": {},
        }
        self._save()
        logger.info("Run manifest created: %s", self.dir)

    def add_metrics(self, name: str, metrics: dict) -> Path:
        """Store a metrics block and also write it as a standalone JSON."""
        self.manifest["metrics"][name] = metrics
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        self.manifest["artifacts"][name] = str(path.relative_to(self.dir))
        self._save()
        return path

    def add_artifact(self, name: str, path: Path | str) -> None:
        self.manifest["artifacts"][name] = str(path)
        self._save()

    def path(self, filename: str) -> Path:
        return self.dir / filename

    def _save(self) -> None:
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def latest() -> Path | None:
        if not RUNS_ROOT.exists():
            return None
        runs = sorted(RUNS_ROOT.glob("run_*"), reverse=True)
        return runs[0] if runs else None
