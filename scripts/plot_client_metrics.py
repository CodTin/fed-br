import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from common.const import CLIENT_METRICS_PATTERN, CLIENT_PLOTS_DIR, FINAL_MODEL_DIR


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def find_latest_metrics_file(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob(CLIENT_METRICS_PATTERN))
    if not candidates:
        raise SystemExit(f"No metrics files found in {output_dir}")
    return candidates[-1]


def plot_metrics(records: list[dict[str, Any]], output_dir: Path) -> None:
    by_phase_metric: dict[str, dict[str, dict[int, list[tuple[int, float]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for record in records:
        phase = str(record.get("phase", "unknown"))
        round_number = int(record.get("round", 0))
        client_id = int(record.get("client_id", -1))
        metrics: dict[str, Any] = record.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, bool):
                metric_value = float(int(value))
            elif isinstance(value, (int, float)):
                metric_value = float(value)
            else:
                continue
            by_phase_metric[phase][metric_name][client_id].append(
                (round_number, metric_value)
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    for phase, metrics in by_phase_metric.items():
        for metric_name, client_series in metrics.items():
            plt.figure(figsize=(10, 6))
            for client_id, series in sorted(client_series.items()):
                series_sorted = sorted(series, key=lambda item: item[0])
                rounds = [item[0] for item in series_sorted]
                values = [item[1] for item in series_sorted]
                plt.plot(
                    rounds,
                    values,
                    marker="o",
                    linewidth=1.2,
                    label=f"Client {client_id}",
                )
            plt.title(f"{phase.title()} {metric_name}")
            plt.xlabel("Round")
            plt.ylabel(metric_name)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="best", fontsize="small")
            plt.tight_layout()
            filename = f"{phase}_{metric_name}.png"
            plt.savefig(output_dir / filename, dpi=150)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-client metrics from Flower JSONL logs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to client metrics JSONL (defaults to latest run in outputs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(CLIENT_PLOTS_DIR),
        help="Directory for output PNG files",
    )
    args = parser.parse_args()

    input_path = args.input or find_latest_metrics_file(Path(f"../{FINAL_MODEL_DIR}"))
    records = load_records(input_path)
    if not records:
        raise SystemExit(f"No records found in {args.input}")
    plot_metrics(records, args.output_dir)


if __name__ == "__main__":
    main()
