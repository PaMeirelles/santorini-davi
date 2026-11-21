"""Where the Davi pipeline reads and writes. Override any of these with environment variables.

Code lives in this repository; data, checkpoints and logs do not (they are multi-GB).

    SANTORINI_MEDIATOR  santoriniMediator checkout (game rules + match databases)
    DAVI_DATA           root for generated data (default: <repo>/data, git-ignored)
    DAVI_CORPUS         Raven-format corpus: samples.bin, splits, raw DB snapshot
    DAVI_JOKER_RUN      Joker run dir: context.bin, dataset.json, checkpoints, logs
"""
import os
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
REPO = TRAINING.parent
ENGINE_SRC = REPO / "engine" / "src"

MEDIATOR = Path(os.environ.get("SANTORINI_MEDIATOR", r"C:\Users\rafae\PycharmProjects\santoriniMediator"))
DATA = Path(os.environ.get("DAVI_DATA", REPO / "data"))
CORPUS = Path(os.environ.get("DAVI_CORPUS", DATA / "raven-corpus"))
JOKER_RUN = Path(os.environ.get("DAVI_JOKER_RUN", DATA / "joker-run"))
