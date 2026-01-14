import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ClientMetricLogger:
    """Append per-client metrics to a JSON Lines file."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        round_number: int,
        phase: str,
        client_id: int,
        metrics: Mapping[str, Any],
    ) -> None:
        record = {
            "round": round_number,
            "phase": phase,
            "client_id": client_id,
            "metrics": dict(metrics),
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(record, ensure_ascii=False)}\n")
