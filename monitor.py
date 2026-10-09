import argparse
import json
from collections import Counter
import numpy as np
from recsys.config import ROOT
from recsys.utils import save_json


def summarize(path):
    if not path.is_file():
        raise FileNotFoundError("Request log missing. Run demo.py or main.py first.")
    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not events:
        raise ValueError("Request log is empty.")
    successful = [event for event in events if event["status"] == "ok"]
    latencies = [event["query_latency_ms"] for event in events]
    slots = sum((event["returned_k"] for event in successful))
    return {
        "scope": "Actual local request events only; not production traffic.",
        "request_count": len(events),
        "successful_requests": len(successful),
        "error_rate": round(1 - len(successful) / len(events), 6),
        "latency_p50_ms": round(float(np.quantile(latencies, 0.5)), 4),
        "latency_p95_ms": round(float(np.quantile(latencies, 0.95)), 4),
        "latency_p99_ms": round(float(np.quantile(latencies, 0.99)), 4),
        "fallback_slot_rate": (
            round(sum((event["fallback_items"] for event in successful)) / slots, 6)
            if slots
            else None
        ),
        "short_list_requests": sum(
            (event["returned_k"] < event["requested_k"] for event in successful)
        ),
        "methods": dict(Counter((event["method"] for event in events))),
        "model_versions": dict(Counter((event["model_version"] for event in events))),
        "business_metrics": {
            "ctr": None,
            "watch_start_conversion": None,
            "watch_minutes": None,
            "retention": None,
        },
        "business_metric_status": "No real impression, click, playback or retention events were collected.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", type=lambda value: ROOT / value, default=ROOT / "logs/requests.jsonl"
    )
    parser.add_argument("--output", default=str(ROOT / "results/monitoring.json"))
    args = parser.parse_args()
    result = summarize(args.log)
    save_json(result, args.output)
    print(json.dumps(result, indent=2))
