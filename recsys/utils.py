import hashlib
import json
import socket
from pathlib import Path


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_network():

    def reject(*args, **kwargs):
        raise RuntimeError("Network access is disabled during local execution.")

    socket.socket.connect = reject
    socket.socket.connect_ex = reject
