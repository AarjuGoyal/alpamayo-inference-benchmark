import yaml
from core import load_model, measure_latency, prepare_inputs, measure_error, load_dataset
import argparse
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "baseline.yaml",
        help="Path to config file"
    )
    return parser.parse_args()

def output_results(label, results):
    print(f"Results for {label}")
    for k, v in results.items():
        print(f"{k}: {v:.2f}ms\n")

if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    model, processor = load_model(**config["model"])
    inputs, data = load_dataset(processor,config["inputs"]["clip_id"])
    results = measure_latency(model,inputs, data, **config["benchmark"])
    output_results(label=config["label"], results=results)
    # measure_error(pred_xyz, pred_rot, extra, data)

