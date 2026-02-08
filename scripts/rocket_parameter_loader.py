import json
from pathlib import Path


def load_parameters():
    """Load rocket parameters from rocket-info/parameters.json."""
    params_path = Path(__file__).resolve().parents[1] / "rocket-info" / "parameters.json"
    with params_path.open() as f:
        return json.load(f)
