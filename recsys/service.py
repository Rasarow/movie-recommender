import json
import time
import uuid
from datetime import datetime, timezone
from .config import ROOT


def request(
    model,
    method,
    user_id=None,
    top_k=10,
    liked_movie_ids=(),
    preferred_genre=None,
    blocked_movie_ids=(),
    log_path=None,
):
    started = time.perf_counter()
    event = {
        "request_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "model_version": model.metadata["model_version"],
        "requested_k": top_k,
    }
    try:
        result = model.recommend(
            method, user_id, top_k, liked_movie_ids, preferred_genre, blocked_movie_ids
        )
        event.update(
            status="ok",
            returned_k=result["returned_k"],
            fallback_items=result["fallback_item_count"],
            known_user=result["known_training_user"],
        )
        result["request_id"] = event["request_id"]
        result["query_latency_ms"] = round((time.perf_counter() - started) * 1000, 4)
        return result
    except (ValueError, RuntimeError) as error:
        event.update(status="error", error_type=type(error).__name__)
        raise
    finally:
        event["query_latency_ms"] = round((time.perf_counter() - started) * 1000, 4)
        path = log_path or ROOT / "logs/requests.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
