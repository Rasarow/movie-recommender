from recsys.config import ROOT, METHODS
from recsys.model import MovieModel
from recsys.service import request
from recsys.utils import block_network, save_json
from monitor import summarize
from audit import audit


def demo():
    block_network()
    model = MovieModel.load()
    log = ROOT / "logs/demo_requests.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")
    user = int(model.user_ids[0])
    results = [request(model, method, user_id=user, log_path=log) for method in METHODS]
    results.append(request(model, "item_cf", log_path=log))
    results.append(request(model, "genre", preferred_genre="Sci-Fi", log_path=log))
    results.append(request(model, "content", liked_movie_ids=[1, 260], log_path=log))
    save_json({"offline_network_blocked": True, "requests": results}, ROOT / "results/demo.json")
    save_json(summarize(log), ROOT / "results/monitoring.json")
    audit(model)
    print("Four methods and new-user examples completed. Results: results/demo.json")


if __name__ == "__main__":
    demo()
