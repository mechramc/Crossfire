"""Decision logging for CROSSFIRE-X AutoPilot."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from crossfire.autopilot.policy import ExecutionPolicy
from crossfire.autopilot.query_classifier import QueryClass


@dataclass(frozen=True)
class DecisionRecord:
    """Serializable record of one AutoPilot decision."""

    query_class: QueryClass
    selected_policy: ExecutionPolicy
    was_exploration: bool
    ucb_scores: dict[str, float | None]
    tokens_per_second: float
    tokens_per_watt: float
    ttft_ms: float
    acceptance_rate: float
    reward: float
    execution_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        """Convert the record to a JSON-serializable dictionary."""

        data = asdict(self)
        data["query_class"] = self.query_class.name
        data["selected_policy"] = self.selected_policy.name
        return data


class DecisionLogger:
    """Append-only JSONL logger for AutoPilot decisions."""

    def __init__(self, path: Path) -> None:
        """Initialize the logger target path."""

        self.path = path

    def log(self, record: DecisionRecord) -> None:
        """Append one decision record as JSONL."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
