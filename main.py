import argparse
import json
import sys
from recsys.config import METHODS
from recsys.model import MovieModel
from recsys.service import request
from recsys.utils import save_json, block_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--method", choices=METHODS, required=True)
    recommend.add_argument("--user", type=int, help="MovieLens user ID; omitted means a new user.")
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--liked-movie", type=int, action="append", default=[])
    recommend.add_argument("--preferred-genre")
    recommend.add_argument("--blocked-movie", type=int, action="append", default=[])
    recommend.add_argument("--output")
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--search", default="")
    catalog.add_argument("--limit", type=int, default=20)
    commands.add_parser("genres")
    args = parser.parse_args()
    block_network()
    try:
        model = MovieModel.load()
        if args.command == "genres":
            result = model.genres
        elif args.command == "catalog":
            if not 1 <= args.limit <= 100:
                raise ValueError("Catalog limit must be between 1 and 100.")
            selected = model.movies[
                model.movies.title.str.contains(args.search, case=False, regex=False)
            ]
            result = selected.head(args.limit).to_dict(orient="records")
        else:
            result = request(
                model,
                args.method,
                args.user,
                args.k,
                args.liked_movie,
                args.preferred_genre,
                args.blocked_movie,
            )
            if args.output:
                save_json(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
