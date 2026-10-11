"""
Project-level carbon tracking helpers for model training.

All project carbon data lives under ``<project>/carbon_tracking`` so it
survives deletion of model run folders.
"""

import copy
import json
import os
import re
import time
import traceback
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


LEDGER_VERSION = 1


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "training"


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON values in configs to simple serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        backup_path = path.with_suffix(path.suffix + f".corrupt_{int(time.time())}")
        try:
            path.replace(backup_path)
        except OSError:
            pass
        return copy.deepcopy(default)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _empty_ledger() -> Dict[str, Any]:
    return {
        "version": LEDGER_VERSION,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "totals": {
            "runs": 0,
            "tracked_runs": 0,
            "duration_s": 0.0,
            "energy_kwh": 0.0,
            "co2eq_g": 0.0,
            "co2eq_kg": 0.0,
        },
        "runs": [],
    }


def _recompute_totals(ledger: Dict[str, Any]) -> None:
    runs = ledger.get("runs", [])
    duration_s = 0.0
    energy_kwh = 0.0
    co2eq_g = 0.0
    tracked_runs = 0

    for run in runs:
        carbon = run.get("carbon", {})
        actual = carbon.get("actual", {})
        run_energy = actual.get("energy_kwh")
        run_co2 = actual.get("co2eq_g")
        run_duration = actual.get("duration_s")

        if isinstance(run_energy, (int, float)) or isinstance(run_co2, (int, float)):
            tracked_runs += 1
        if isinstance(run_energy, (int, float)):
            energy_kwh += float(run_energy)
        if isinstance(run_co2, (int, float)):
            co2eq_g += float(run_co2)
        if isinstance(run_duration, (int, float)):
            duration_s += float(run_duration)

    ledger["totals"] = {
        "runs": len(runs),
        "tracked_runs": tracked_runs,
        "duration_s": round(duration_s, 6),
        "energy_kwh": round(energy_kwh, 9),
        "co2eq_g": round(co2eq_g, 6),
        "co2eq_kg": round(co2eq_g / 1000.0, 9),
    }


class CarbonTrainingRun(AbstractContextManager):
    """Best-effort carbontracker wrapper with a persistent project ledger."""

    def __init__(
        self,
        project_path: Path,
        model_kind: str,
        training_name: str,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[Path] = None,
    ):
        self.project_path = Path(project_path)
        self.model_kind = model_kind
        self.training_name = training_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id = f"{timestamp}_{_slugify(model_kind)}_{_slugify(training_name)}"

        self.root_dir = self.project_path / "carbon_tracking"
        self.runs_dir = self.root_dir / "runs"
        self.run_dir = self.runs_dir / self.run_id
        self.log_dir = self.run_dir / "carbontracker_logs"
        self.ledger_path = self.root_dir / "carbon_usage.json"
        self.summary_path = self.run_dir / "carbon_summary.json"

        self.config = _json_safe(config or {})
        self.model_path = self._relative_path(model_path) if model_path else None
        self.started_at = None
        self.ended_at = None
        self._start_time = None
        self._tracker = None
        self._tracker_available = False
        self._tracker_started = False
        self._error = None

    def __enter__(self) -> "CarbonTrainingRun":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        status = "failed" if exc_type else "completed"
        error_message = None
        if exc_value is not None:
            error_message = "".join(traceback.format_exception_only(exc_type, exc_value)).strip()
        self.finish(status=status, error=error_message)
        return False

    def start(self) -> None:
        self.started_at = _now_iso()
        self._start_time = time.monotonic()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._write_run(status="running")

        try:
            from carbontracker.tracker import CarbonTracker

            self._tracker = CarbonTracker(
                epochs=1,
                epochs_before_pred=0,
                monitor_epochs=1,
                log_dir=str(self.log_dir),
                log_file_prefix=f"{self.run_id}_",
                verbose=1,
                ignore_errors=True,
            )
            self._tracker_available = True
            self._tracker.epoch_start()
            self._tracker_started = True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._tracker = None
            self._tracker_started = False
            self._write_run(status="running")

    def finish(
        self,
        status: str = "completed",
        model_path: Optional[Path] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        if model_path is not None:
            self.model_path = self._relative_path(model_path)
        self.ended_at = _now_iso()

        if self._tracker is not None and self._tracker_started:
            try:
                self._tracker.epoch_end()
            except Exception as exc:
                self._error = self._merge_errors(self._error, f"epoch_end: {type(exc).__name__}: {exc}")
            try:
                self._tracker.stop()
            except Exception as exc:
                self._error = self._merge_errors(self._error, f"stop: {type(exc).__name__}: {exc}")
            finally:
                self._tracker_started = False

        if error:
            self._error = self._merge_errors(self._error, error)

        run = self._build_run(status=status, metrics=metrics)
        self._upsert_run(run)
        _atomic_write_json(self.summary_path, run)
        return run

    def metrics_for_final_report(self) -> Dict[str, float]:
        run = _read_json(self.summary_path, {})
        carbon = run.get("carbon", {})
        actual = carbon.get("actual", {})
        metrics = {}

        for output_key, source_key in [
            ("carbon_energy_kwh", "energy_kwh"),
            ("carbon_co2eq_g", "co2eq_g"),
            ("carbon_duration_s", "duration_s"),
        ]:
            value = actual.get(source_key)
            if isinstance(value, (int, float)):
                metrics[output_key] = float(value)

        ledger = _read_json(self.ledger_path, _empty_ledger())
        totals = ledger.get("totals", {})
        total_energy = totals.get("energy_kwh")
        total_co2 = totals.get("co2eq_g")
        if isinstance(total_energy, (int, float)):
            metrics["project_carbon_energy_kwh_total"] = float(total_energy)
        if isinstance(total_co2, (int, float)):
            metrics["project_carbon_co2eq_g_total"] = float(total_co2)

        return metrics

    def _build_run(self, status: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        actual = self._parse_actual_consumption()
        if actual.get("duration_s") is None and self._start_time is not None:
            actual["duration_s"] = round(time.monotonic() - self._start_time, 3)

        carbon = {
            "available": self._tracker_available,
            "actual": actual,
            "log_dir": self._relative_path(self.log_dir),
        }
        if self._error:
            carbon["error"] = self._error

        return {
            "run_id": self.run_id,
            "model_kind": self.model_kind,
            "training_name": self.training_name,
            "status": status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model_path": self.model_path,
            "config": self.config,
            "metrics": _json_safe(metrics or {}),
            "carbon": carbon,
        }

    def _write_run(self, status: str) -> None:
        self._upsert_run(self._build_run(status=status))

    def _upsert_run(self, run: Dict[str, Any]) -> None:
        ledger = _read_json(self.ledger_path, _empty_ledger())
        ledger.setdefault("version", LEDGER_VERSION)
        ledger.setdefault("created_at", _now_iso())
        ledger.setdefault("runs", [])

        runs = ledger["runs"]
        for idx, existing in enumerate(runs):
            if existing.get("run_id") == self.run_id:
                runs[idx] = run
                break
        else:
            runs.append(run)

        ledger["updated_at"] = _now_iso()
        _recompute_totals(ledger)
        _atomic_write_json(self.ledger_path, ledger)
        _atomic_write_json(self.summary_path, run)

    def _parse_actual_consumption(self) -> Dict[str, Optional[float]]:
        actual = {"duration_s": None, "energy_kwh": None, "co2eq_g": None}
        parser_error = None
        try:
            from carbontracker import parser

            logs = parser.parse_all_logs(log_dir=str(self.log_dir))
            if logs:
                parsed_actual = logs[-1].get("actual", {})
                actual["duration_s"] = _as_float(parsed_actual.get("duration (s)"))
                actual["energy_kwh"] = _as_float(parsed_actual.get("energy (kWh)"))
                actual["co2eq_g"] = _as_float(parsed_actual.get("co2eq (g)"))
        except Exception as exc:
            parser_error = f"{type(exc).__name__}: {exc}"

        # CarbonTracker's parser rejects a directory when auxiliary log files
        # make its input/output log counts differ.  The output log still holds
        # the final measurements, so recover them directly in that case.
        if actual["energy_kwh"] is None or actual["co2eq_g"] is None:
            fallback = self._parse_output_log()
            for key, value in fallback.items():
                if actual[key] is None:
                    actual[key] = value

        if parser_error and actual["energy_kwh"] is None and actual["co2eq_g"] is None:
            self._error = self._merge_errors(self._error, f"parse: {parser_error}")
        return actual

    def _parse_output_log(self) -> Dict[str, Optional[float]]:
        actual = {"duration_s": None, "energy_kwh": None, "co2eq_g": None}
        output_logs = sorted(
            self.log_dir.glob("*_output.log"),
            key=lambda path: path.stat().st_mtime,
        )
        if not output_logs:
            return actual
        try:
            content = output_logs[-1].read_text(errors="replace")
        except OSError:
            return actual

        energy = re.search(r"^\s*Energy:\s*([0-9.eE+-]+)\s*kWh\s*$", content, re.MULTILINE)
        co2eq = re.search(r"^\s*CO2eq:\s*([0-9.eE+-]+)\s*g\s*$", content, re.MULTILINE | re.IGNORECASE)
        duration = re.search(r"^\s*Time:\s*(\d+):(\d+):(\d+(?:\.\d+)?)\s*$", content, re.MULTILINE)
        if energy:
            actual["energy_kwh"] = _as_float(energy.group(1))
        if co2eq:
            actual["co2eq_g"] = _as_float(co2eq.group(1))
        if duration:
            hours, minutes, seconds = duration.groups()
            actual["duration_s"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        return actual

    def _relative_path(self, path: Path) -> str:
        """Return a portable path relative to the project directory."""
        return os.path.relpath(str(Path(path)), start=str(self.project_path))

    @staticmethod
    def _merge_errors(existing: Optional[str], new_error: str) -> str:
        if not existing:
            return new_error
        return f"{existing}; {new_error}"


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
