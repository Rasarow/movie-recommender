Movie recommendation system

This project recommends previously unrated films using MovieLens small data. It implements genre-based content filtering, item-based collaborative filtering, rating-count popularity and a genre-specific adjusted-rating rule.

The tested environment is Windows x64 with Python 3.12. Inference is local and no API key or GPU is required.

Installation

Run this command with Python 3.12 from the project directory.

python setup.py

Setup creates .venv, installs the dependencies, downloads and verifies MovieLens, fits the training models and runs the evaluation and demo. Initial preparation needs internet access.

Usage

run.cmd demo.py
run.cmd main.py recommend --method content --user 1
run.cmd main.py recommend --method item_cf --user 1
run.cmd main.py recommend --method popular --user 1
run.cmd main.py recommend --method genre --preferred-genre Sci-Fi
run.cmd main.py recommend --method content --liked-movie 1 --liked-movie 260
run.cmd main.py catalog --search "Star Wars"
run.cmd evaluate.py
run.cmd monitor.py --log logs/demo_requests.jsonl

MovieLens user IDs are sample users. Omitting --user means a new user. Explicit liked-film IDs are excluded from the returned list. --k changes list length, --blocked-movie excludes a film, and --output saves the recommendation JSON. Use the catalog command to find valid film IDs.

The selected genre controls the genre rule. It can initialize an empty content profile. Item-based collaborative filtering uses co-like history and does not use genre selection.

Files

recsys contains the model, data and metric functions. prepare.py obtains the verified dataset. fit.py creates local artifacts. main.py returns recommendations. evaluate.py compares the methods on later ratings. demo.py creates examples and request metrics. audit.py demonstrates cold start and an isolated popularity manipulation. REPORT.txt explains results, limitations and production monitoring.

Raw data, fitted artifacts, logs and the Python environment are excluded from version control. Setup recreates them. Measured summaries are included in results. MovieLens usage conditions apply; see data/DATA_SOURCE.json and the official dataset documentation.
