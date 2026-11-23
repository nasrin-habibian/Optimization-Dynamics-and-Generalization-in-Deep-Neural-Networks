"""Command-line entry point."""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dlopt.experiments import run_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    run_experiment(config, project_root=ROOT)

if __name__ == "__main__":
    main()
